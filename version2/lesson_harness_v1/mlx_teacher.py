"""Compressed offline annotation teacher, isolated from the research runtime.

Conversion and labeling remain subject to the same 12 GiB reserves. No remote
quantized checkpoint is substituted: derive it from the authenticated local files.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .acquire_teacher import REPO,REVISION
from .local_teacher import verify_checkpoint,parse_response
from .rubrics import RUBRICS,prompt,FORMAT_SYSTEM


def runtime():
    isolated=Path(__file__).resolve().parents[1]/'.local/lesson-harness-v1/runtime-packages'
    sys.path.insert(0,str(isolated))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')


def convert(checkpoint,output):
    checkpoint,output=map(Path,(checkpoint,output));guard(output)
    if output.exists():raise ValueError('preserve prior conversion; choose a new directory')
    # This pinned 3B checkpoint has ~3.2B weights. Four-bit affine groups of 64
    # with float16 scales/biases take ~1.7 GiB; retain a 2 GiB derivative budget.
    # The watchdog still checks actual free space throughout conversion.
    if shutil.disk_usage(output.parent).free<14*1024**3:
        raise RuntimeError('insufficient external headroom for quantized derivative')
    verify_checkpoint(checkpoint);runtime()
    import mlx.core as mx
    from mlx_lm.convert import convert as mlx_convert
    from importlib.metadata import version
    mx.set_default_device(mx.cpu)
    started=time.monotonic()
    mlx_convert(hf_path=str(checkpoint),mlx_path=str(output),quantize=True,q_group_size=64,q_bits=4,
                q_mode='affine',dtype='float16',upload_repo=None,trust_remote_code=False)
    guard(output)
    meta={'schema':'perception-local-quantized-teacher-v1','complete':True,'parent_repo':REPO,'parent_revision':REVISION,
        'parent_acquisition_sha256':file_sha256(checkpoint/'acquisition.json'),'quantization':{'bits':4,'group_size':64,'mode':'affine'},
        'unquantized_dtype':'float16','conversion_device':'cpu','elapsed_seconds':time.monotonic()-started,
        'runtime':{n:version(n) for n in ['mlx','mlx-lm','mlx-metal','transformers']},
        'files':{p.name:file_sha256(p) for p in output.iterdir() if p.is_file()},
        'exact_full_precision_equivalence_claimed':False,'participant_data_used':False,'remote_upload':False}
    (output/'quantization.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Quantized teacher saved and verified',flush=True)


def label(checkpoint,packets,output):
    checkpoint,packets,output=map(Path,(checkpoint,packets,output));guard(output);runtime()
    meta=json.loads((checkpoint/'quantization.json').read_text())
    if (not meta['complete'] or meta['parent_repo']!=REPO or meta['parent_revision']!=REVISION
            or meta['quantization']!={'bits':4,'group_size':64,'mode':'affine'}):raise ValueError('qualified quantized derivative required')
    for name,digest in meta['files'].items():
        guard(output)
        path=(checkpoint/name).resolve()
        if path.parent!=checkpoint.resolve() or file_sha256(path)!=digest:raise ValueError('quantized teacher identity mismatch')
    import mlx.core as mx
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(17)
    model,tokenizer=load(str(checkpoint),tokenizer_config={'trust_remote_code':False},trust_remote_code=False)
    output.mkdir(parents=True,exist_ok=False);results=[]
    for record in json.loads(packets.read_text()):
        for rubric in RUBRICS:
            guard(output);packet=record['packet'];text=prompt(rubric,packet)
            encoded=tokenizer.apply_chat_template([{'role':'system','content':FORMAT_SYSTEM},{'role':'user','content':text}],
                tokenize=True,add_generation_prompt=True,date_string='04 Nov 2026')
            if len(encoded)>4096:raise ValueError('annotation exceeds fixed input-token limit')
            started=time.monotonic();parts=[];tokens=0
            for response in stream_generate(model,tokenizer,encoded,max_tokens=900,sampler=make_sampler(temp=0)):
                tokens+=1;parts.append(response.text)
                if tokens%8==0:guard(output)
            result={'id':record['id'],'rubric':rubric,'packet_sha256':__import__('hashlib').sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest(),
                'raw_response':''.join(parts),'elapsed_seconds':time.monotonic()-started,'generated_tokens':tokens,'status':'invalid'}
            try:result.update(status='valid',response=parse_response(result['raw_response'],packet))
            except (ValueError,TypeError,KeyError) as error:result['error']=type(error).__name__+': '+str(error)
            results.append(result);(output/'responses.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            print(record['id']+' '+rubric+': '+result['status'],flush=True)
    report={'schema':'perception-local-rubric-run-v1','complete':True,'backend':'mlx-affine-4bit-group64','repo':REPO,'revision':REVISION,
        'quantization_sha256':file_sha256(checkpoint/'quantization.json'),'packets_sha256':file_sha256(packets),
        'responses_sha256':file_sha256(output/'responses.json'),'rubrics_sha256':file_sha256(Path(__file__).with_name('rubrics.py')),
        'runner_sha256':file_sha256(Path(__file__)),'seed':17,'decoding':'greedy','max_new_tokens':900,
        'format_system_message':True,'chat_template_date':'04 Nov 2026',
        'validated_responses':sum(r['status']=='valid' for r in results),'total_responses':len(results),
        'participant_data_sent_to_cloud':False,'interpretation':'synthetic judgments; not human ground truth'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


def watch(action,checkpoint,output,packets=None):
    output=Path(output);guard(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('preserve existing run')
    log_path=output.with_name(output.name+'-runtime.log')
    args=[sys.executable,'-B','-m','lesson_harness_v1.mlx_teacher',action,str(Path(checkpoint).resolve()),str(output.resolve()),'--worker']
    if packets:args+=['--packets',str(Path(packets).resolve())]
    with log_path.open('x') as log:
        process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parents[1])
        try:
            while process.poll() is None:guard(output);time.sleep(1)
            if process.returncode:raise RuntimeError('MLX teacher process failed; inspect '+str(log_path))
        except BaseException as error:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            output.with_name(output.name+'-failure.json').write_text(json.dumps({'complete':False,'error':str(error),'participant_data_sent_to_cloud':False})+'\n')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['convert','label','ordinal']);p.add_argument('checkpoint',type=Path)
    p.add_argument('output',type=Path);p.add_argument('--packets',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.action in {'label','ordinal'} and a.packets is None:p.error('--packets required for label')
    if not a.worker:watch(a.action,a.checkpoint,a.output,a.packets)
    elif a.action=='convert':convert(a.checkpoint,a.output)
    elif a.action=='ordinal':
        from .ordinal_teacher import label as ordinal_label
        ordinal_label(a.checkpoint,a.packets,a.output)
    else:label(a.checkpoint,a.packets,a.output)
