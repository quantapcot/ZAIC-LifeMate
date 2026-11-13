"""Bounded QLoRA training on the prepared TRAIN-only first-action dataset."""
import argparse
import json
import random
import time
from pathlib import Path
from scripts.evaluate_action_agent import WORKTREE, ROOT
from src.toolcall.evaluation.heldout import Heldout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--model-dir',default='outputs/action_sft/model');p.add_argument('--output-dir',required=True)
    p.add_argument('--resume-from',help='Local adapter checkpoint with matching optimizer/RNG state');p.add_argument('--steps',type=int,default=50);p.add_argument('--smoke-steps',type=int,default=2);p.add_argument('--grad-accum',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.steps<a.smoke_steps or a.smoke_steps<1 or a.grad_accum<1:raise ValueError('invalid step configuration')
    out=(WORKTREE/a.output_dir).resolve();base=(WORKTREE/'outputs/action_sft').resolve()
    if base not in out.parents:raise ValueError('output must be under outputs/action_sft')
    dataset=(WORKTREE/a.dataset).resolve();manifest=json.loads((dataset/'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('source_partition')!='train' or manifest.get('model')!='Qwen/Qwen3-4B-Instruct-2507':raise ValueError('invalid dataset provenance')
    rows=lambda path:[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    fit=rows(dataset/'fit.jsonl');dev=rows(dataset/'dev.jsonl')
    if not fit or not dev or set(r['id'] for r in fit)&set(r['id'] for r in dev):raise ValueError('invalid fit/dev')
    source=Heldout(ROOT/'train/train.jsonl',ROOT/'train/train_tools_catalog.json',WORKTREE.parent/'toolloop-synthesis/outputs/toolloop/splits',partition='train')
    if manifest.get('dataset_sha256')!=source.manifest['dataset_sha256'] or manifest.get('catalog_sha256')!=source.manifest['catalog_sha256']:raise ValueError('dataset identity mismatch')
    if not {r['id'] for r in fit+dev}<=source.ids:raise ValueError('non-TRAIN example')
    if set(r['id'] for r in fit)!=set(manifest['selected_ids']['fit']) or set(r['id'] for r in dev)!=set(manifest['selected_ids']['dev']):raise ValueError('dataset IDs do not match manifest')
    for row in fit+dev:
        if len(row['input_ids'])!=len(row['labels']) or row['labels'].count(-100)!=row['prompt_tokens'] or row['labels'][:row['prompt_tokens']]!=[-100]*row['prompt_tokens'] or row['labels'][row['prompt_tokens']:]!=row['input_ids'][row['prompt_tokens']:]:raise ValueError('invalid loss mask')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig,PeftConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training
    from src.toolcall.training.checkpoint import fingerprint,save_resume,load_resume
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);out.mkdir(parents=True,exist_ok=False)
    path=str((WORKTREE/a.model_dir).resolve());tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(path,quantization_config=q,device_map={'':0},torch_dtype=dtype,local_files_only=True,attn_implementation='sdpa')
    if (model.config.model_type,model.config.hidden_size,model.config.num_hidden_layers,model.config.vocab_size)!=('qwen3',2560,36,151936):raise RuntimeError('unexpected Qwen3-4B architecture')
    model.config.use_cache=False;model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj'])

    if a.resume_from:
        previous=(WORKTREE/a.resume_from).resolve()
        if base not in previous.parents or not (previous/'resume_state.pt').is_file():raise ValueError('resume checkpoint must be local and complete')
        pc=PeftConfig.from_pretrained(str(previous),local_files_only=True)
        if Path(pc.base_model_name_or_path).resolve()!=Path(path) or pc.r!=8 or pc.lora_alpha!=16 or pc.lora_dropout!=0.05 or set(pc.target_modules)!={'q_proj','v_proj'}:raise ValueError('resume adapter mismatch')
        model=PeftModel.from_pretrained(model,str(previous),is_trainable=True,local_files_only=True)
    else:model=get_peft_model(model,config)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);total=sum(p.numel() for p in model.parameters())
    if trainable==0:raise RuntimeError('no trainable parameters')
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=0.0001)
    rng=random.Random(a.seed);order=list(range(len(fit)));rng.shuffle(order);cursor=0;log=[];scores=[];first_step=1
    model_weights=sorted(Path(path).glob('*.safetensors'))
    if not model_weights:raise ValueError('local safetensor weights required')
    identity=fingerprint([dataset/'dataset_manifest.json',dataset/'fit.jsonl',dataset/'dev.jsonl',Path(path)/'config.json',*model_weights],{'seed':a.seed,'grad_accum':a.grad_accum,'base':path,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'lr':0.0001,'lora_rank':8,'dtype':str(dtype)})
    if a.resume_from:
        saved=load_resume(previous/'resume_state.pt',optimizer,rng,identity);order=saved['order'];cursor=saved['cursor'];log=saved['log'];scores=saved['scores'];first_step=saved['step']+1
        if a.steps<first_step:raise ValueError('resume target must exceed saved step')
    def batch(row):
        return {'input_ids':torch.tensor([row['input_ids']],device='cuda',dtype=torch.long),'labels':torch.tensor([row['labels']],device='cuda',dtype=torch.long),'attention_mask':torch.ones((1,len(row['input_ids'])),device='cuda',dtype=torch.long)}
    def evaluate():
        model.eval();loss=[]
        with torch.inference_mode():
            for row in dev[:min(8,len(dev))]:loss.append(float(model(**batch(row)).loss))
        model.train();return sum(loss)/len(loss)
    model.train();optimizer.zero_grad(set_to_none=True)
    for step in range(first_step,a.steps+1):
        t=time.perf_counter();loss_total=0.
        for _ in range(a.grad_accum):
            if cursor==len(order):rng.shuffle(order);cursor=0
            row=fit[order[cursor]];cursor+=1
            loss=model(**batch(row)).loss
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss_total+=float(loss.detach());(loss/a.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
        item={'step':step,'loss':loss_total/a.grad_accum,'seconds':time.perf_counter()-t,'peak_gpu_bytes':torch.cuda.max_memory_allocated()};log.append(item);print(json.dumps(item),flush=True)
        if step in (a.smoke_steps,a.steps):
            score=evaluate();ckpt=out/('adapter-step-'+str(step));model.save_pretrained(ckpt);scores.append({'step':step,'dev_loss':score,'path':str(ckpt),'dev_samples':min(8,len(dev))});save_resume(ckpt/'resume_state.pt',optimizer,rng,{'step':step,'order':order,'cursor':cursor,'log':log,'scores':scores},identity);print(json.dumps({'checkpoint_step':step,'dev_loss':score}),flush=True)
        (out/'training_log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    selected=min(scores,key=lambda x:x['dev_loss']);summary={'base_model':'Qwen/Qwen3-4B-Instruct-2507','model_dir':path,'dataset_manifest':str(dataset/'dataset_manifest.json'),'steps_completed':len(log),'smoke_steps':a.smoke_steps,'gradient_accumulation':a.grad_accum,'seed':a.seed,'lora_rank':8,'lora_alpha':16,'lora_dropout':0.05,'target_modules':['q_proj','v_proj'],'learning_rate':0.0001,'quantization':'4-bit NF4 double quant','compute_dtype':str(dtype),'trainable_parameters':trainable,'total_parameters':total,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-start,'checkpoints':scores,'selected_adapter':selected['path'],'resumed_from':str(previous) if a.resume_from else None,'final_train_loss':log[-1]['loss'],'fit_examples':len(fit),'dev_examples':len(dev),'seen_training_examples':a.steps*a.grad_accum,'tokenizer_eos_id':tokenizer.eos_token_id}
    (out/'train_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
