"""Local Transformers/PEFT backend compatible with ActionAgent.generate_json."""
import contextlib
import json
import time
from pathlib import Path
from src.toolcall.llm.ollama_client import PromptBudgetError,_check_output,InvalidModelOutput,TransportError


class TransformersActionModel:
    def __init__(self,model,tokenizer,use_adapter=True,max_new_tokens=256,context_limit=2048):
        self.model=model;self.tokenizer=tokenizer;self.use_adapter=use_adapter;self.max_new_tokens=max_new_tokens;self.context_limit=context_limit
        self.timeout=30
        if not hasattr(model,'peft_config') or not model.peft_config:raise ValueError('loaded PEFT adapter required for paired evaluation')
        self.model.eval()
    def generate_json(self,system_prompt,user_prompt,output_schema):
        import torch
        messages=[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
        return self.generate_ids(ids,output_schema)
    def generate_ids(self,ids,output_schema):
        import torch
        if not torch.is_tensor(ids):ids=torch.tensor([ids],dtype=torch.long)
        if ids.ndim==1:ids=ids.unsqueeze(0)
        count=ids.shape[-1]
        budget={'estimated_input_tokens':count,'reserved_output_tokens':self.max_new_tokens,'context_tokens':self.context_limit,'estimation':'exact_tokenizer'}
        if count+self.max_new_tokens>self.context_limit:raise PromptBudgetError(budget)
        ids=ids.to('cuda');start=time.perf_counter()
        active=contextlib.nullcontext() if self.use_adapter else self.model.disable_adapter()
        with active,torch.inference_mode():
            output=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=self.max_new_tokens,do_sample=False,pad_token_id=self.tokenizer.eos_token_id,eos_token_id=self.tokenizer.eos_token_id,use_cache=True,max_time=max(0.1,self.timeout))
        if time.perf_counter()-start>=self.timeout and int(output[0,-1])!=self.tokenizer.eos_token_id:raise TransportError('generation exceeded deadline',code='TIMEOUT')
        text=self.tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True).strip();self.last_text=text
        try:value=json.loads(text,parse_constant=lambda value: (_ for _ in ()).throw(ValueError('invalid JSON constant')))
        except (ValueError,TypeError) as exc:raise InvalidModelOutput('invalid JSON',code='INVALID_JSON') from exc
        _check_output(value,output_schema)
        return value,{'output_tokens':output.shape[-1]-count,'latency_seconds':time.perf_counter()-start,'input_budget':budget}


def load_paired(base_dir,adapter_dir):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,PeftConfig
    base=Path(base_dir).resolve();adapter=Path(adapter_dir).resolve()
    if not (adapter/'adapter_config.json').is_file() or not (adapter/'adapter_model.safetensors').is_file():raise ValueError('adapter files missing')
    pc=PeftConfig.from_pretrained(str(adapter),local_files_only=True)
    if Path(pc.base_model_name_or_path).resolve()!=base:raise ValueError('adapter base model mismatch')
    tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(str(base),quantization_config=q,device_map={'':0},torch_dtype=dtype,attn_implementation='sdpa',local_files_only=True)
    if (model.config.model_type,model.config.hidden_size,model.config.num_hidden_layers,model.config.vocab_size)!=('qwen3',2560,36,151936):raise ValueError('unexpected Qwen3-4B architecture')
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
    if not model.active_adapters or not any('lora_' in n for n,_ in model.named_parameters()):raise ValueError('adapter not active')
    model.eval();return model,tokenizer
