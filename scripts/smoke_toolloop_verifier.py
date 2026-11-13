"""Synthetic-only verifier controls; no competition records or ToolEnv."""
import argparse
import json
from src.toolcall.data import ToolRegistry
from src.toolcall.data.contracts import GroundTruth, Metadata
from src.toolcall.llm import OllamaClient
from src.toolcall.verification import SemanticVerifier


def run_controls(client):
    raw,usage=client.generate_json('Return JSON only.','Return {"ok": true}.',{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']})
    if raw != {'ok':True}: raise ValueError('JSON smoke failed')
    catalog={'schema_version':'synthetic-fixture','apps':[{'id':'fixture','group':'fixture','tools':[
        {'name':'fixture.remind','description':'Create a reminder with the supplied reminder text.',
         'parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}]}]}
    registry=ToolRegistry(catalog)
    ground=GroundTruth('simple',['fixture.remind'],['fixture.remind'],Metadata('fixture','synthetic'))
    verify=SemanticVerifier(client)
    cases=[('rubric',2,'Nhắc tôi mua sữa',None,True),
           ('intent_mismatch',2,'Xóa lời nhắc mua sữa',None,False),
           ('hallucinated_argument',3,'Nhắc tôi mua sữa',[{'name':'fixture.remind','arguments':{'text':'invented'}}],False),
           ('uncertainty',2,'Nhắc tôi',None,False)]
    result={'json':{'passed':True,'latency_seconds':usage['latency_seconds']},'controls':{}}
    for name,stage,query,calls,expected in cases:
        verdict=verify.verify(stage,ground,registry,query=query,calls=calls)
        observed=verdict['verdict']['verdict']
        met = verdict['passed'] if expected else (not verdict['passed'] and observed != 'pass')
        result['controls'][name]={'observed':observed,'policy_passed':verdict['passed'],
                                  'expected_behavior':expected,'met_expectation':met,
                                  'latency_seconds':verdict['usage'].get('latency_seconds')}
    result['all_controls_met']=all(x['met_expectation'] for x in result['controls'].values())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verifier-model',default='qwen3:8b-q4_K_M')
    parser.add_argument('--generator-model',default='qwen3:4b-instruct-2507-q4_K_M')
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if args.verifier_model==args.generator_model:raise ValueError('distinct models required')
    generator=OllamaClient(model=args.generator_model,keep_alive=0,confirm_unloaded=True,timeout=args.timeout)
    client=OllamaClient(model=args.verifier_model,temperature=0,keep_alive=0,think=False,confirm_unloaded=True,timeout=args.timeout)
    gd,vd=generator.model_digest(),client.model_digest()
    if not gd or not vd or gd==vd:raise ValueError('distinct local model digests required')
    generator.unload();client.unload()
    result=run_controls(client);result['verifier_model']=client.model;result['verifier_digest']=vd
    print(json.dumps(result,ensure_ascii=True,indent=2))
    if not result['all_controls_met']:raise SystemExit(1)


if __name__=='__main__':main()
