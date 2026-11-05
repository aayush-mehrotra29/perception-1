"""Offline fixed-checkpoint rubric inference with a separate resource watchdog."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .acquire_teacher import REPO,REVISION,FILES
from .rubrics import prompt,validate_response,RUBRICS


def verify_checkpoint(checkpoint):
    checkpoint=Path(checkpoint);receipt=json.loads((checkpoint/'acquisition.json').read_text())
    if not receipt['complete'] or receipt['repo']!=REPO or receipt['revision']!=REVISION or set(receipt['files'])!=set(FILES):
        raise ValueError('complete pinned local instruction checkpoint required')
    for name,record in receipt['files'].items():
        guard(checkpoint)
        if file_sha256(checkpoint/name)!=record['sha256']:raise ValueError('teacher checkpoint identity changed')
    return receipt


def parse_response(text,packet):
    text=text.strip()
    if text.startswith('```json\n') and text.endswith('```'):text=text[8:-3].strip()
    result=json.loads(text)
    return validate_response(result,[item['id'] for item in packet['evidence']])


def worker(checkpoint,packets,output):
    # Applied before importing the model libraries; no provider clients are used.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    # The installed Transformers loader otherwise converts many tensors on MPS
    # concurrently. Serial conversion also bounds transient memory during loading.
    os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    checkpoint,packets,output=map(Path,(checkpoint,packets,output));guard(output)
    isolated_runtime=Path(__file__).resolve().parents[1]/'.local/lesson-harness-v1/runtime-packages'
    if isolated_runtime.exists():sys.path.insert(0,str(isolated_runtime))
    receipt=verify_checkpoint(checkpoint)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList
    torch.set_num_threads(1);torch.manual_seed(17)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    print('loading pinned local teacher on '+device,flush=True)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,
        dtype=torch.float16,device_map={'':device},attn_implementation='eager')
    model.eval();guard(output)
    class ResourceStop(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            if input_ids.shape[-1]%8==0:guard(output)
            return False
    records=json.loads(packets.read_text());results=[]
    for record in records:
        for rubric in RUBRICS:
            guard(output);packet=record['packet'];text=prompt(rubric,packet)
            encoded=tokenizer.apply_chat_template([{'role':'user','content':text}],add_generation_prompt=True,return_tensors='pt',return_dict=True).to(device)
            if encoded['input_ids'].shape[-1]>4096:raise ValueError('local annotation exceeds 4096 input tokens')
            started=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**encoded,max_new_tokens=900,do_sample=False,use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([ResourceStop()]))
            generated_tokens=generated[0,encoded['input_ids'].shape[-1]:]
            response=tokenizer.decode(generated_tokens,skip_special_tokens=True)
            result={'id':record['id'],'rubric':rubric,'packet_sha256':__import__('hashlib').sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest(),
                'raw_response':response,'elapsed_seconds':time.monotonic()-started,'generated_tokens':len(generated_tokens),
                'status':'invalid','interpretation':'synthetic rubric judgment, not observed learner state'}
            try:result.update(status='valid',response=parse_response(response,packet))
            except (ValueError,TypeError,KeyError) as error:result['error']=type(error).__name__+': '+str(error)
            results.append(result)
            (output/'responses.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            print(record['id']+' '+rubric+': '+result['status'],flush=True)
    manifest={'schema':'perception-local-rubric-run-v1','complete':True,'repo':REPO,'revision':REVISION,
        'checkpoint_receipt_sha256':file_sha256(checkpoint/'acquisition.json'),'packets_sha256':file_sha256(packets),
        'responses_sha256':file_sha256(output/'responses.json'),'rubrics_sha256':file_sha256(Path(__file__).with_name('rubrics.py')),
        'runner_sha256':file_sha256(Path(__file__)),'device':device,'dtype':'float16','max_new_tokens':900,
        'asynchronous_checkpoint_loading':False,
        'decoding':'greedy','seed':17,'participant_data_sent_to_cloud':False,
        'validated_responses':sum(r['status']=='valid' for r in results),'total_responses':len(results)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def run(checkpoint,packets,output):
    output=Path(output);guard(output);output.mkdir(parents=True,exist_ok=False)
    args=[sys.executable,'-B','-m','lesson_harness_v1.local_teacher','worker',str(Path(checkpoint).resolve()),str(Path(packets).resolve()),str(output.resolve())]
    with (output/'runtime.log').open('w') as log:
        process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parents[1])
        try:
            while process.poll() is None:
                guard(output);time.sleep(1)
            if process.returncode:raise RuntimeError('local teacher failed; see runtime.log')
        except BaseException as error:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            (output/'failure.json').write_text(json.dumps({'complete':False,'error':str(error),
                'partial_responses_preserved':True,'participant_data_sent_to_cloud':False})+'\n')
            raise
    return json.loads((output/'manifest.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker']);p.add_argument('checkpoint',type=Path)
    p.add_argument('packets',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    (run if a.action=='run' else worker)(a.checkpoint,a.packets,a.output)
