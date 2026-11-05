"""Single authorized dispatch; preserves its call ID and conservative reservation."""
from pathlib import Path
import json
import time

from brain.credentials import load_credentials
from brain.credits import CreditLedger
from brain.edugage_targets import guard
from .cortical_replay import prepare_request, validate_request, materialize, file_sha


def run(*, resume_undispatched=False):
    base = Path('../.local/brain-runs')
    sources = {'healing_radiation': base/'healing_radiation-native-brain-v1',
               'xray_diffraction': base/'xray_diffraction-native-brain-v1',
               'atmosphere_pressure_composition': base/'atmosphere_pressure_composition-native-brain-v2',
               'rocket': base/'rocket-native-brain-v4',
               'example_audio': base/'full-audio-qualification-v1/tribe',
               'example_video': base/'full-video-qualification-v1/tribe'}
    job = Path('.local/numerical-brain-v1/native-audio-replay-job-r1')
    output = Path('.local/numerical-brain-v1/native-audio-replay-r1')
    if output.exists(): raise FileExistsError('Existing output must be recovered')
    if resume_undispatched:
        if not (job/'reserved.json').is_file() or (job/'call.json').exists() or (job/'result.zip').exists():
            raise ValueError('Only an unused saved reservation may resume local preflight')
    elif job.exists(): raise FileExistsError('Existing dispatch must be recovered, never retried automatically')
    guard(job); request = prepare_request(sources, 'runs/tribe-hidden-pilot'); byte_count = validate_request(request)
    job.mkdir(parents=True, exist_ok=resume_undispatched)
    plan = {'sources': {k: str(v.resolve()) for k,v in sources.items()}, 'input_bytes': byte_count,
        'batch_count': sum(len(r['dense']) for r in request['sources']),
        'source_manifest_sha256': {r['name']: r['source_manifest_sha256'] for r in request['sources']},
        'reservation_microdollars': 700000, 'one_dispatch_only': True, 'retries': 0,
        'timeout_seconds': 300, 'startup_timeout_seconds': 300, 'cpu_hard_limit': 2, 'memory_hard_limit_MiB': 16384,
        'gpu': 'A100-80GB', 'pricing_source': 'https://modal.com/pricing',
        'rates_USD_per_second': {'gpu': .000694, 'cpu_total': .0000262, 'memory_total': .00003552},
        '600_second_compute_estimate_USD': .453432, 'remainder_reserved_for_build_egress_and_overhead_USD': .246568,
        'provider_actual_charge_not_yet_known': True,
        'source_code_sha256': {name: file_sha(Path('numerical_brain_v1')/name)
            for name in ['cortical_replay.py','cloud_replay.py','replay_launch.py']}}
    plan_name = 'plan-resume-undispatched.json' if resume_undispatched else 'plan.json'
    if (job/plan_name).exists(): raise FileExistsError('Preserve earlier explicit resume')
    (job/plan_name).write_text(json.dumps(plan, indent=2)+'\n')
    ledger = CreditLedger(Path('../.local/brain-credits.sqlite3'))
    identifier = 'numerical-brain-v1-dense-replay-r1'
    if not resume_undispatched:
        ledger.reserve(identifier, 'modal', 700000)
        (job/'reserved.json').write_text(json.dumps({'identifier': identifier, 'snapshot': ledger.snapshot()}, indent=2)+'\n')
    elif json.loads((job/'reserved.json').read_text())['identifier'] != identifier:
        raise ValueError('saved reservation identifier differs')
    load_credentials(Path('../.local/brain-credentials.json'))
    from .cloud_replay import app, replay
    started = time.monotonic()
    try:
        with app.run(detach=True):
            call = replay.spawn(request)
            (job/'call.json').write_text(json.dumps({'function_call_id': call.object_id,
                'reservation_identifier': identifier, 'can_retrieve_without_new_inference': True}, indent=2)+'\n')
            print(json.dumps({'function_call_id': call.object_id, 'status': 'dispatched'}), flush=True)
            archive = call.get()
            # Preserve paid result before any local unpacking or validation.
            (job/'result.zip').write_bytes(archive)
            receipt = materialize(archive, request, output)
            (job/'completed.json').write_text(json.dumps({'elapsed_seconds': time.monotonic()-started,
                'result_sha256': file_sha(job/'result.zip'), 'output': str(output.resolve()),
                'receipt': receipt, 'cost_status': 'full reservation retained pending actual billing'}, indent=2)+'\n')
            print(json.dumps(receipt), flush=True)
    except BaseException as error:
        (job/('failure-resume.json' if resume_undispatched else 'failure.json')).write_text(json.dumps({'exception_type': type(error).__name__,
            'message': str(error)[:2000], 'automatic_retry': False,
            'reservation_retained': True, 'elapsed_seconds': time.monotonic()-started}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--resume-undispatched', action='store_true'); a = p.parse_args()
    run(resume_undispatched=a.resume_undispatched)
