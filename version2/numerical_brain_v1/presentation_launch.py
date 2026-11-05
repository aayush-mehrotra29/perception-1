"""Stage public media separately, then make one explicitly authorized GPU call."""
import hashlib
import json
from pathlib import Path
import re
import time

from brain.credentials import load_credentials
from brain.credits import CreditLedger
from brain.edugage_targets import guard
from brain.modal_client import single_gpu_job
from brain.worker_contract import file_sha256
from .presentation_batch import prepare,materialize

ROOT=Path(__file__).resolve().parents[1]
JOB=ROOT/'.local/numerical-brain-v1/narrated-native-job-r1'
INPUT=ROOT/'.local/numerical-brain-v1/narrated-inputs-r1/manifest.json'
OUTPUT=ROOT/'.local/numerical-brain-v1/narrated-native-r1'
STATE=ROOT.parent/'.local'


def stage():
    if JOB.exists():raise ValueError('preserve prior staging; inspect it before recovery')
    guard(JOB);items=prepare(INPUT);JOB.mkdir(parents=True)
    local=JOB/'input';local.mkdir();records=[]
    for item in items:
        name=item['name'];filename=name+('.wav' if name=='text' else '.mp4')
        (local/filename).write_bytes(item['media']);records.append({'name':name,'request':item['request'],'file':filename})
    m={'schema':'perception-staged-narrated-media-v1','records':records}
    (local/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    digest=file_sha256(local/'manifest.json');remote='numerical-presentation-inputs/'+digest
    load_credentials(STATE/'brain-credentials.json')
    import modal
    volume=modal.Volume.from_name('perception-public-model-cache',create_if_missing=False)
    with volume.batch_upload(force=False) as upload:upload.put_directory(local,'/'+remote)
    receipt={'manifest_path':remote+'/manifest.json','manifest_sha256':digest,'media_bytes':sum(len(i['media']) for i in items),
        'input_manifest_sha256':file_sha256(INPUT),'paid_model_calls':0,'participant_data_or_weights_uploaded':False}
    (JOB/'staged.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def run():
    ready=ROOT/'.local/numerical-brain-v1/native-audio-replay-job-r2/completed.json'
    if not ready.is_file():raise ValueError('await corrected replay completion before using shared GPU')
    staged=json.loads((JOB/'staged.json').read_text())
    if file_sha256(INPUT)!=staged['input_manifest_sha256']:raise ValueError('staged original input list changed')
    if any((JOB/name).exists() for name in ('reserved.json','call.json','result.json')) or OUTPUT.exists():
        raise ValueError('prior paid call preserved; never submit automatically again')
    guard(JOB);items=prepare(INPUT);load_credentials(STATE/'brain-credentials.json')
    ledger=CreditLedger(STATE/'brain-credits.sqlite3');identifier='numerical-narrated-three-presentations-r1'
    plan={'reservation_microdollars':3000000,'timeout_seconds':660,'startup_timeout_seconds':180,'cpu_limit':2,
        'memory_limit_MiB':32768,'gpu':'A100-80GB','one_dispatch_only':True,'retries':0,
        'input_bytes':staged['media_bytes'],'RPC_media_or_array_bytes':0,'original_input_manifest_sha256':file_sha256(INPUT),
        'result_transport':'durable Volume archive; small hash/path RPC receipt','source_hashes':{
            n:file_sha256(Path(__file__).with_name(n)) for n in ('cloud_presentations.py','presentation_launch.py','presentation_batch.py')}}
    (JOB/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    started=time.monotonic()
    with single_gpu_job(STATE/'brain-gpu.lock'):
        ledger.reserve(identifier,'modal',3000000)
        (JOB/'reserved.json').write_text(json.dumps({'identifier':identifier,'snapshot':ledger.snapshot()},indent=2)+'\n')
        try:
            from .cloud_presentations import app,presentations
            with app.run(detach=True):
                (JOB/'app.json').write_text(json.dumps({'app_id':app.app_id})+'\n')
                call=presentations.spawn(staged['manifest_path'],staged['manifest_sha256'])
                (JOB/'call.json').write_text(json.dumps({'function_call_id':call.object_id,'app_id':app.app_id,
                    'reservation_identifier':identifier,'can_retrieve_without_new_inference':True})+'\n')
                print(json.dumps({'status':'dispatched','function_call_id':call.object_id}),flush=True)
                result=call.get();(JOB/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            remote=result['archive_path']
            if (not re.fullmatch(r'narrated-numerical-archives/[0-9a-f]{64}.zip',remote)
                    or not 0<result['archive_bytes']<=256*1024**2):raise ValueError('invalid bounded result archive path')
            import modal
            volume=modal.Volume.from_name('perception-public-model-cache',create_if_missing=False)
            archive=JOB/'result.zip'
            with archive.open('xb') as stream:
                for chunk in volume.read_file('/'+remote):guard(JOB);stream.write(chunk)
            if archive.stat().st_size!=result['archive_bytes'] or file_sha256(archive)!=result['archive_sha256']:
                raise ValueError('downloaded result archive identity mismatch')
            receipt=materialize(archive.read_bytes(),items,OUTPUT)
            report={'complete':True,'output':str(OUTPUT),'elapsed_seconds':time.monotonic()-started,'receipt':receipt,
                'full_reservation_retained_pending_provider_billing':True}
            (JOB/'completed.json').write_text(json.dumps(report,indent=2)+'\n');return report
        except BaseException as error:
            (JOB/'failure.json').write_text(json.dumps({'exception_type':type(error).__name__,'message':str(error)[:2000],
                'automatic_retry':False,'paid_outputs_may_be_recovered_without_new_inference':True,
                'reservation_retained':True},indent=2)+'\n');raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage-only',action='store_true');a=parser.parse_args()
    print(json.dumps(stage() if a.stage_only else run(),indent=2))
