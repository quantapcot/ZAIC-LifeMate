"""Aggregate-only lexical leakage warnings; no user text is returned."""
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
import re
from .split import query_key


def leakage_audit(records, splits):
    partition = {rid: name for name, ids in splits.items() for rid in ids}
    if len(partition) != sum(map(len, splits.values())) or set(partition) != {r["id"] for r in records}:
        raise ValueError("invalid split assignments")
    rows = sorted(((r["id"], query_key(r["audio_text"])) for r in records), key=lambda x: x[0])
    inverted = defaultdict(list)
    for i, (_, text) in enumerate(rows):
        for token in set(text.split()): inverted[token].append(i)
    candidates = set()
    for members in inverted.values():
        for a, b in combinations(members, 2):
            if partition[rows[a][0]] != partition[rows[b][0]]: candidates.add((a, b))
    exact = template = similar = 0
    for a, b in sorted(candidates):
        x, y = rows[a][1], rows[b][1]
        if not x or not y: continue
        if x == y:
            exact += 1
            continue
        # Number substitution is only a weak template heuristic.
        tx, ty = re.sub(r"\d+", "#", x), re.sub(r"\d+", "#", y)
        if tx == ty:
            template += 1
        xs, ys = set(x.split()), set(y.split())
        if len(xs & ys) / len(xs | ys) >= .75 and SequenceMatcher(None, x, y, autojunk=False).ratio() >= .88:
            similar += 1
    return {"cross_split_exact_normalized_pairs": exact,
            "cross_split_number_template_pairs": template,
            "cross_split_lexically_similar_pairs": similar,
            "note": "heuristic counts can overlap; no semantic deduplication or reassignment"}
