"""Explicit single corrected launch using durable public-content volume transport."""
from pathlib import Path
import io
import json
import time
import zipfile
from brain.credentials import load_credentials
from brain.credits import CreditLedger
from brain.edugage_targets import guard
from .cortical_replay import prepare_request, stage_request, materialize, file_sha


def run():
    base = Path('../.local/brain-runs'); root = Path('.local/numerical-brain-v1')
    job = root/'native-audio-replay-job-r2'; output = root/'native-audio-replay-r2'
    if job.exists() or output.exists(): raise FileExistsError('existing staged job must be recovered, not resubmitted')
    prior = json.loads((root/'native-audio-replay-job-r1/call-graph.json').read_text())
    if not all(r['status'] == 'GENERIC_STATUS_FAILURE' for r in prior['inputs']): raise ValueError('prior call not terminal')
    apps = json.loads((root/'native-audio-replay-job-r1/apps-after.json').read_text())['apps']
    failed = next(a for a in apps if a['app_id'] == 'ap-roCXDFXjR8YPZGrXW20RSt')
    if failed['state'] != 'APP_STATE_STOPPED' or failed.get('n_running_tasks', 0): raise ValueError('prior worker still active')
    guard(job); job.mkdir(parents=True)
    sources = {'healing_radiation': base/'healing_radiation-native-brain-v1',
               'xray_diffraction': base/'xray_diffraction-native-brain-v1',
               'atmosphere_pressure_composition': base/'atmosphere_pressure_composition-native-brain-v2',
               'rocket': base/'rocket-native-brain-v4',
               'example_audio': base/'full-audio-qualification-v1/tribe',
               'example_video': base/'full-video-qualification-v1/tribe'}
    request = prepare_request(sources, 'runs/tribe-hidden-pilot')
    stage = job/'input'; identity = stage_request(request, stage)
    identifier = 'numerical-brain-v1-dense-replay-staged-r2'; remote_id = 'numerical-brain-replay-r2'
    remote = '/numerical-brain-replay/' + remote_id
    ledger = CreditLedger(Path('../.local/brain-credits.sqlite3')); ledger.reserve(identifier, 'modal', 700000)
    plan = {'schema': 'perception-staged-replay-launch-v1', 'explicit_manual_correction': True,
        'prior_provider_failure_preserved': True, 'prior_app_stopped': True,
        'change': 'stage authenticated public-content inputs and outputs on existing Volume; RPC carries only ID/hash; enable network for platform transport',
        'model_and_tolerance_unchanged': True, 'automatic_paid_retry': False, 'new_reservation_microdollars': 700000,
        'old_reservation_retained': True, 'ledger': ledger.snapshot(), 'remote_id': remote_id, **identity,
        'no_foundation_downloads': True, 'no_participant_or_EEG_weight_uploads': True,
        'code_sha256': {n: file_sha(Path('numerical_brain_v1')/n) for n in ['cloud_replay.py','cortical_replay.py','replay_staged_launch.py']}}
    (job/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    load_credentials(Path('../.local/brain-credentials.json'))
    import modal
    from .cloud_replay import app, replay_staged
    volume = modal.Volume.from_name('perception-public-model-cache', create_if_missing=False)
    started = time.monotonic()
    try:
        with volume.batch_upload(force=False) as upload: upload.put_directory(stage, remote+'/input')
        (job/'uploaded.json').write_text(json.dumps({'remote_input': remote+'/input', **identity})+'\n')
        print(json.dumps({'status': 'public_content_staged', **identity}), flush=True)
        with app.run(detach=True):
            (job/'app.json').write_text(json.dumps({'app_id': app.app_id})+'\n')
            call = replay_staged.spawn(remote_id, identity['request_sha256'])
            (job/'call.json').write_text(json.dumps({'function_call_id': call.object_id, 'app_id': app.app_id})+'\n')
            print(json.dumps({'function_call_id': call.object_id, 'status': 'dispatched'}), flush=True)
            result = call.get(); (job/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        expected = {'receipt.json'}
        for source in request['sources']:
            expected.add(source['name']+'/manifest.json')
            expected.update(source['name']+'/'+r['file'].replace('dense-', 'hidden-') for r in source['dense'])
        if set(result['files']) != expected: raise ValueError('unexpected remote output list')
        downloads = job/'download'; downloads.mkdir()
        for name, checksum in result['files'].items():
            local = downloads/name; local.parent.mkdir(parents=True, exist_ok=True)
            with local.open('xb') as f:
                for chunk in volume.read_file(remote+'/output/'+name): f.write(chunk)
            if file_sha(local) != checksum: raise ValueError('volume output checksum differs')
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as z:
            for name in result['files']: z.write(downloads/name, name)
        report = materialize(stream.getvalue(), request, output)
        (job/'completed.json').write_text(json.dumps({'elapsed_seconds': time.monotonic()-started,
            'output': str(output.resolve()), 'receipt': report, 'full_reservation_retained_pending_provider_billing': True}, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    except BaseException as error:
        (job/'failure.json').write_text(json.dumps({'exception_type': type(error).__name__, 'message': str(error)[:2000],
            'automatic_retry': False, 'remote_outputs_may_be_recovered_without_new_inference': True}, indent=2)+'\n')
        raise


if __name__ == '__main__': run()
