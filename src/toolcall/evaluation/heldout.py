"""Read-only validation accessor. Labels are decoded only after inference finishes."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def inputs_only(line):
    decoder = json.JSONDecoder()
    i = 0
    def whitespace(i):
        while i < len(line) and line[i].isspace(): i += 1
        return i
    i = whitespace(i)
    if line[i] != '{': raise ValueError('record is not an object')
    i += 1; result = {}; saw_label = False
    while True:
        i = whitespace(i)
        if line[i] == '}': break
        key, i = decoder.raw_decode(line, i); i = whitespace(i)
        if line[i] != ':': raise ValueError('invalid object')
        i = whitespace(i + 1)
        if key == 'label':
            saw_label = True
            if line[i] != '{': raise ValueError('invalid label')
            stack = []; string = False; escaped = False
            while i < len(line):
                c = line[i]; i += 1
                if string:
                    if escaped: escaped = False
                    elif c == '\\': escaped = True
                    elif c == '"': string = False
                elif c == '"': string = True
                elif c in '{[': stack.append(c)
                elif c in '}]':
                    if not stack or (stack.pop(), c) not in (('{','}'),('[',']')): raise ValueError('invalid label nesting')
                    if not stack: break
            if stack or string: raise ValueError('invalid label')
        else:
            value, i = decoder.raw_decode(line, i)
            if key in ('id','audio_text','context','history'): result[key] = value
        i = whitespace(i)
        if line[i] == ',': i += 1; continue
        if line[i] == '}': break
        raise ValueError('invalid object separator')
    if not saw_label: raise ValueError('missing label')
    return result


class Heldout:
    def __init__(self, dataset, catalog, splits):
        self.dataset = Path(dataset); self.catalog = Path(catalog); splits = Path(splits)
        self.manifest = json.loads((splits/'split_manifest.json').read_text(encoding='utf-8'))
        if digest(dataset) != self.manifest['dataset_sha256'] or digest(catalog) != self.manifest['catalog_sha256']: raise ValueError('source hash mismatch')
        parts = {p: json.loads((splits/(p+'_ids.json')).read_text(encoding='utf-8')) for p in ('train','val','test')}
        seen = set()
        for name, ids in parts.items():
            if len(ids) != self.manifest['counts'][name] or len(set(ids)) != len(ids) or seen.intersection(ids): raise ValueError('invalid split')
            seen.update(ids)
        self.ids = set(parts['val']); self.records = {}; all_ids = set()
        with self.dataset.open(encoding='utf-8') as f:
            for line in f:
                row = inputs_only(line); key = row['id']
                if key in all_ids: raise ValueError('duplicate dataset ID')
                all_ids.add(key)
                if key in self.ids: self.records[key] = row
        if seen != all_ids: raise ValueError('split coverage mismatch')
        self._predictions_finished = False

    def finish_predictions(self): self._predictions_finished = True

    def labels(self, ids):
        if not self._predictions_finished: raise RuntimeError('labels unavailable before predictions finish')
        ids = set(ids)
        if not ids <= self.ids: raise ValueError('only validation labels allowed')
        if digest(self.dataset) != self.manifest['dataset_sha256']: raise ValueError('dataset changed')
        labels = {}
        with self.dataset.open(encoding='utf-8') as f:
            for line in f:
                key = inputs_only(line)['id']
                if key in ids: labels[key] = json.loads(line)['label']
        if set(labels) != ids: raise ValueError('missing labels')
        return labels
