"""One externally budgeted, no-retry 300s head-only TRIBE replay job.

Importing this module does not invoke it. Caller must reserve the complete cost
before .remote(), and retain failures without automatically resubmitting them.
"""
import modal
from cloud_hidden import image as native_image

app = modal.App('perception-dense-tribe-replay-v1')
cache = modal.Volume.from_name('perception-public-model-cache', create_if_missing=False)
image = native_image.add_local_python_source('numerical_brain_v1', 'cloud_hidden')


@app.function(image=image, gpu='A100-80GB', cpu=(2, 2), memory=(16384, 16384), timeout=300, startup_timeout=300,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              block_network=True, single_use_containers=True,
              env={'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1'},
              volumes={'/model-cache': cache})
def replay(request):
    import io
    import tempfile
    import zipfile
    from pathlib import Path
    from numerical_brain_v1.cortical_replay import run_replay
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / 'replay'
        run_replay(request, output, Path('/model-cache'))
        files = [p for p in output.rglob('*') if p.is_file()]
        if sum(p.stat().st_size for p in files) > 64 * 1024**2:
            raise ValueError('response bound exceeded')
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as z:
            for p in files: z.write(p, str(p.relative_to(output)))
        return stream.getvalue()


@app.function(image=image, gpu='A100-80GB', cpu=(2, 2), memory=(16384, 16384), timeout=300, startup_timeout=300,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              single_use_containers=True,
              env={'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1'},
              volumes={'/model-cache': cache})
def replay_staged(job_id, request_sha256):
    """Explicitly corrected transport: only an ID/hash enter the RPC."""
    import json
    import re
    from pathlib import Path
    from numerical_brain_v1.cortical_replay import load_staged_request, run_replay, file_sha
    if not re.fullmatch('numerical-brain-replay-r[2-9][0-9]*', job_id): raise ValueError('owned stage identifier required')
    root = Path('/model-cache/numerical-brain-replay')/job_id
    request = load_staged_request(root/'input', request_sha256)
    report = run_replay(request, root/'output', Path('/model-cache'))
    # Commit every paid output before returning a small recoverable receipt.
    cache.commit()
    return {'schema': 'perception-staged-replay-result-v1', 'job_id': job_id,
            'request_sha256': request_sha256, 'receipt_sha256': file_sha(root/'output/receipt.json'),
            'files': {str(p.relative_to(root/'output')): file_sha(p)
                      for p in (root/'output').rglob('*') if p.is_file()}, 'summary': report}
