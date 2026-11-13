"""Safe local optimizer-state checkpoints for deterministic continuation."""
import hashlib
from pathlib import Path


def fingerprint(paths,settings):
    import json
    h=hashlib.sha256(json.dumps(settings,sort_keys=True).encode())
    for path in paths:
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save_resume(path,optimizer,rng,state,identity):
    import torch
    checkpoint={'identity':identity,'optimizer':optimizer.state_dict(),'python_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'state':state}
    path=Path(path);temporary=path.with_suffix('.tmp');torch.save(checkpoint,temporary);temporary.replace(path)


def load_resume(path,optimizer,rng,identity):
    import torch
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('identity')!=identity:raise ValueError('resume identity mismatch')
    optimizer.load_state_dict(checkpoint['optimizer']);rng.setstate(checkpoint['python_rng']);torch.set_rng_state(checkpoint['torch_rng'])
    if checkpoint['cuda_rng']:
        if not torch.cuda.is_available() or len(checkpoint['cuda_rng'])!=torch.cuda.device_count():raise ValueError('resume CUDA mismatch')
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['state']
