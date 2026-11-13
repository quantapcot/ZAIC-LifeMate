"""Deterministic, label-blind lexical retrieval over installed catalog metadata."""
import json
import math
import re
import unicodedata
from collections import Counter

STOP = {'cua','cho','toi','mot','nhung','cac','voi','va','la','trong','den','the','a','an','the','of','to','for','and','is','on','at','tool','cong','cu'}

def tokens(text):
    text = unicodedata.normalize('NFKD', str(text).lower().replace('đ','d'))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return [x for x in re.findall(r'[a-z0-9]+',text) if len(x)>1 and x not in STOP]

class ToolRouter:
    def __init__(self, registry, top_k=8):
        if top_k < 1: raise ValueError('top_k must be positive')
        self.registry=registry; self.top_k=top_k

    def rank(self, available, request, context=None, observations=(), offset=0):
        # Query uses only legitimate request, recent observed results and permitted context.
        query=Counter(tokens(request+' '+json.dumps(context or {},ensure_ascii=False)+' '+json.dumps(list(observations)[-2:],ensure_ascii=False)))
        metadata={}
        for name in available:
            if name == 'sys.ask_missing_info': continue
            tool=self.registry.get_tool(name)
            metadata[name]=Counter(tokens(name.replace('_',' ')*2+' '+tool.get('description','')+' '+json.dumps(tool['parameters'],ensure_ascii=False)))
        df=Counter(t for words in metadata.values() for t in words)
        total=max(len(metadata),1)
        scores={name:sum((1+math.log(1+count))*min(query[t],2)*(1+math.log((total+1)/(df[t]+1))) for t,count in words.items() if t in query) for name,words in metadata.items()}
        ranked=sorted(scores,key=lambda n:(-scores[n],n))
        # A bounded shifted window provides one alternative shortlist when a finish is uncertain.
        selected=ranked[offset:offset+self.top_k]
        if 'sys.ask_missing_info' in available: selected.append('sys.ask_missing_info')
        return selected, {n:round(scores[n],3) for n in selected if n in scores}
