"""One explicitly reviewed timeout recovery; never an automatic paid retry."""
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
from .presentation_launch import ROOT,STATE,INPUT


def run(stopped_receipt):
    prior=ROOT/'.local/numerical-brain-v1/narrated-native-job-r1'
    job=ROOT/'.local/numerical-brain-v1/narrated-native-recovery-job-r1'
    output=ROOT/'.local/numerical-brain-v1/narrated-native-r2'
    if job.exists() or output.exists():raise ValueError('prior recovery preserved; no resubmission')
    failure=json.loads((prior/'failure.json').read_text())
    if failure.get('exception_type')!='FunctionTimeoutError':raise ValueError('reviewed timeout receipt required')
    stopped=json.loads(Path(stopped_receipt).read_text())
    if (stopped.get('app_id')!='ap-HOIQcP754dFgN0SKxJZsy7' or not stopped.get('stopped')
            or stopped.get('running_tasks')!=0):raise ValueError('original app must be stopped with no running tasks')
    status=json.loads((prior/'remote-status-final.json').read_text())
    manifests=[r['path'] for r in status['files'] if r['path'].endswith('/manifest.json')]
    if len(manifests)!=1 or not manifests[0].endswith('/native/text/manifest.json'):
        raise ValueError('review missing-source inventory before recovery')
    staged=json.loads((prior/'staged.json').read_text());items=prepare(INPUT)
    if staged['input_manifest_sha256']!=file_sha256(INPUT):raise ValueError('input identity changed')
    guard(job);job.mkdir(parents=True);load_credentials(STATE/'brain-credentials.json')
    ledger=CreditLedger(STATE/'brain-credits.sqlite3');identifier='numerical-narrated-deliberate-recovery-r1'
    rates={'gpu':.000694,'cpu_per_core':.0000131,'ram_per_GiB':.00000222}
    maximum_compute=(1200+180+2)*(rates['gpu']+8*rates['cpu_per_core']+64*rates['ram_per_GiB'])
    plan={'explicit_recovery_authorized':True,'reservation_microdollars':1500000,'original_3000000_hold_retained':True,
        'reason':'Original660-second combined job timed out after text; committed text reused, missing image/PDF only.',
        'inference_and_checkpoint_unchanged':True,'timeout_seconds':1200,'startup_timeout_seconds':180,'shutdown_seconds':2,
        'cpu_hard_limit':8,'memory_hard_limit_MiB':65536,'gpu':'A100-80GB','retries':0,'max_containers':1,
        'rates_USD_per_second':rates,'maximum_compute_estimate_USD':maximum_compute,'pricing_source':'https://modal.com/pricing',
        'region_multiplier':None,'nonpreemptible_multiplier':None,'overhead_reserve_USD':1.5-maximum_compute,
        'original_stopped_receipt_sha256':file_sha256(Path(stopped_receipt)),
        'final_inventory_sha256':file_sha256(prior/'remote-status-final.json'),
        'source_code_sha256':{n:file_sha256(Path(__file__).with_name(n)) for n in ('cloud_presentations_recovery.py','presentation_recovery_launch.py')}}
    (job/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');started=time.monotonic()
    with single_gpu_job(STATE/'brain-gpu.lock'):
        ledger.reserve(identifier,'modal',1500000)
        (job/'reserved.json').write_text(json.dumps({'identifier':identifier,'ledger':ledger.snapshot()},indent=2)+'\n')
        try:
            from .cloud_presentations_recovery import app,recover
            with app.run(detach=True):
                (job/'app.json').write_text(json.dumps({'app_id':app.app_id})+'\n')
                call=recover.spawn(staged['manifest_path'],staged['manifest_sha256'])
                (job/'call.json').write_text(json.dumps({'function_call_id':call.object_id,'app_id':app.app_id,
                    'reservation_identifier':identifier,'can_retrieve_without_new_inference':True})+'\n')
                print(json.dumps({'status':'explicit_recovery_dispatched','function_call_id':call.object_id}),flush=True)
                result=call.get();(job/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            remote=result['archive_path']
            if not re.fullmatch(r'narrated-numerical-recovery-archives/[0-9a-f]{64}.zip',remote) or not 0<result['archive_bytes']<=256*1024**2:
                raise ValueError('bounded recovery result required')
            import modal
            volume=modal.Volume.from_name('perception-public-model-cache',create_if_missing=False);archive=job/'result.zip'
            with archive.open('xb') as stream:
                for chunk in volume.read_file('/'+remote):guard(job);stream.write(chunk)
            if archive.stat().st_size!=result['archive_bytes'] or file_sha256(archive)!=result['archive_sha256']:
                raise ValueError('recovery archive identity mismatch')
            receipt=materialize(archive.read_bytes(),items,output)
            report={'complete':True,'output':str(output),'elapsed_seconds':time.monotonic()-started,'receipt':receipt,
                'full_reservation_retained_pending_provider_billing':True}
            (job/'completed.json').write_text(json.dumps(report,indent=2)+'\n');return report
        except BaseException as error:
            (job/'failure.json').write_text(json.dumps({'exception_type':type(error).__name__,'message':str(error)[:2000],
                'automatic_retry':False,'reservation_retained':True,'prior_artifacts_preserved':True},indent=2)+'\n');raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stopped-receipt',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.stopped_receipt),indent=2))
