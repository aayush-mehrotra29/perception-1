"""Externally budgeted native inference for three authenticated presentations.

Importing does not dispatch work. The caller must reserve the full bounded GPU
cost in the shared ledger first. Only supplied public/permitted media are inputs.
"""
import modal
from cloud_hidden import image as native_image,cache,hf

app=modal.App('perception-narrated-numerical-v1')
image=native_image.add_local_python_source('numerical_brain_v1','cloud_hidden')


@app.function(image=image,gpu='A100-80GB',cpu=(2,2),memory=(32768,32768),timeout=660,startup_timeout=180,
    retries=0,max_containers=1,min_containers=0,scaledown_window=2,single_use_containers=True,
    volumes={'/model-cache':cache},secrets=[hf])
def presentations(staged_manifest,staged_manifest_sha256):
    import gc
    import hashlib
    import io
    import json
    import os
    import re
    import tempfile
    import time
    import zipfile
    from pathlib import Path
    import torch
    from brain.worker_contract import PublicMediaRequest,validate_media,file_sha256
    from brain.remote_inference import run_tribe
    from numerical_brain_v1.cortical_replay import prepare_request,run_replay
    started=time.monotonic()
    if not isinstance(staged_manifest,str) or not re.fullmatch(r'numerical-presentation-inputs/[0-9a-f]{64}/manifest.json',staged_manifest):
        raise ValueError('staged public-media manifest path required')
    cache.reload();input_path=Path('/model-cache')/staged_manifest
    if file_sha256(input_path)!=staged_manifest_sha256:raise ValueError('staged manifest identity changed')
    staged=json.loads(input_path.read_text())
    if set(staged)!={'schema','records'} or staged['schema']!='perception-staged-narrated-media-v1':raise ValueError('invalid staged inputs')
    items=[]
    for row in staged['records']:
        if set(row)!={'name','request','file'} or not re.fullmatch(r'(text.wav|image.mp4|pdf.mp4)',row['file']):raise ValueError('unexpected staged member')
        media=input_path.parent/row['file'];req=PublicMediaRequest.model_validate(row['request']);validate_media(media,req)
        if req.size_bytes>32*1024**2:raise ValueError('single presentation transport bound exceeded')
        items.append({'name':row['name'],'request':req.model_dump(),'media':media.read_bytes()})
    if (not isinstance(items,list) or [i.get('name') for i in items]!=['text','image','pdf']
            or any(set(i)!={'name','media','request'} for i in items)
            or sum(len(i['media']) for i in items)>64*1024**2):raise ValueError('fixed bounded three-content input required')
    os.environ.update(HF_HOME='/model-cache/huggingface',TORCH_HOME='/model-cache/torch',
        UV_CACHE_DIR='/tmp/perception-uv',UV_LINK_MODE='copy',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory() as scratch:
        batch_key=hashlib.sha256(json.dumps([i['request'] for i in items],sort_keys=True).encode()).hexdigest()
        root=Path('/model-cache/narrated-numerical-v1')/batch_key
        root.mkdir(parents=True,exist_ok=True);native=root/'native';native.mkdir(exist_ok=True)
        for item in items:
            if time.monotonic()-started>480:raise TimeoutError('native extraction time budget exhausted; no retry')
            request=PublicMediaRequest.model_validate(item['request'])
            expected_kind='audio' if item['name']=='text' else 'video'
            if request.media_kind!=expected_kind or len(item['media'])!=request.size_bytes:
                raise ValueError('presentation media type or length mismatch')
            media_root=Path('/model-cache/public-content')/request.sha256;media_root.mkdir(parents=True,exist_ok=True)
            media=media_root/('input.wav' if expected_kind=='audio' else 'input.mp4')
            if not media.exists():media.write_bytes(item['media'])
            validate_media(media,request)
            target=native/item['name']
            if target.exists():
                # Keep interrupted directories intact: they need deliberate
                # recovery, never an automatic paid retry inside this worker.
                raise ValueError('prior presentation batch artifact exists; inspect recovery explicitly')
            run_tribe(media,request,target,Path('/model-cache'))
            cache.commit();gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'native_complete':item['name'],'elapsed_seconds':time.monotonic()-started}),flush=True)
        # Reuse the exact freshly extracted text/audio tensors from each visual
        # presentation. A fresh native audio-only run qualifies replay first.
        if time.monotonic()-started>540:raise TimeoutError('insufficient remaining replay budget; no retry')
        req=prepare_request({'image':native/'image','pdf':native/'pdf'},native/'text')
        replay=root/'audio-replay';receipt=run_replay(req,replay,Path('/model-cache'))
        for record in req['sources']:
            for row in record['dense']:(replay/record['name']/row['file']).write_bytes(row['blob'])
        proof={'schema':'perception-narrated-native-batch-v1','complete':True,'elapsed_seconds':time.monotonic()-started,
            'sources':{i['name']:i['request']['sha256'] for i in items},'replay_qualified':receipt['passed'],
            'cloud_result_cache_key':batch_key,
            'participant_data_received':False,'EEG_weights_received':False,'paid_retries':0,
            'native_and_EEG_audio_features_share_presented_media':True,
            'native_manifests':{i['name']:file_sha256(native/i['name']/'manifest.json') for i in items}}
        (root/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
        cache.commit()
        files=[p for p in root.rglob('*') if p.is_file()]
        if sum(p.stat().st_size for p in files)>256*1024**2:raise ValueError('response size bound exceeded')
        archives=Path('/model-cache/narrated-numerical-archives');archives.mkdir(exist_ok=True)
        archive_path=archives/(batch_key+'.zip')
        with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_STORED) as archive:
            for path in files:archive.write(path,str(path.relative_to(root)))
        cache.commit()
        return {'archive_path':str(archive_path.relative_to('/model-cache')),'archive_sha256':file_sha256(archive_path),
            'archive_bytes':archive_path.stat().st_size,'report':proof}
