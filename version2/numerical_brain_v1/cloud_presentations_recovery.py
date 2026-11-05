"""One deliberate recovery, preserving the original interrupted native captures."""
import modal
from cloud_hidden import image as native_image,cache,hf

app=modal.App('perception-narrated-recovery-v1')
image=native_image.add_local_python_source('numerical_brain_v1','cloud_hidden')


@app.function(image=image,gpu='A100-80GB',cpu=(8,8),memory=(65536,65536),timeout=1200,startup_timeout=180,
    retries=0,max_containers=1,min_containers=0,scaledown_window=2,single_use_containers=True,
    volumes={'/model-cache':cache},secrets=[hf])
def recover(staged_manifest,staged_manifest_sha256):
    import gc,hashlib,json,os,re,shutil,time,zipfile
    from pathlib import Path
    import torch
    from brain.worker_contract import PublicMediaRequest,validate_media,file_sha256
    from brain.remote_inference import run_tribe
    from numerical_brain_v1.cortical_replay import prepare_request,run_replay,_validate_manifest
    started=time.monotonic()
    if not re.fullmatch(r'numerical-presentation-inputs/[0-9a-f]{64}/manifest.json',staged_manifest):raise ValueError('fixed staged source path required')
    cache.reload();source=Path('/model-cache')/staged_manifest
    if file_sha256(source)!=staged_manifest_sha256:raise ValueError('staged input manifest changed')
    data=json.loads(source.read_text())
    if data.get('schema')!='perception-staged-narrated-media-v1' or [r['name'] for r in data['records']]!=['text','image','pdf']:
        raise ValueError('three ordered authenticated presentations required')
    rows=data['records'];batch_key=hashlib.sha256(json.dumps([r['request'] for r in rows],sort_keys=True).encode()).hexdigest()
    prior=Path('/model-cache/narrated-numerical-v1')/batch_key
    root=Path('/model-cache/narrated-numerical-recovery-v1')/batch_key
    if root.exists():raise ValueError('existing recovery must be inspected; no automatic rerun')
    root.mkdir(parents=True);native=root/'native';native.mkdir();reused=[];computed=[]
    os.environ.update(HF_HOME='/model-cache/huggingface',TORCH_HOME='/model-cache/torch',
        UV_CACHE_DIR='/tmp/perception-uv',UV_LINK_MODE='copy',OMP_NUM_THREADS='1')
    for row in rows:
        if set(row)!={'name','request','file'} or row['file']!=row['name']+('.wav' if row['name']=='text' else '.mp4'):
            raise ValueError('unexpected staged media path')
        req=PublicMediaRequest.model_validate(row['request']);media=source.parent/row['file'];validate_media(media,req)
        if req.size_bytes>32*1024**2:raise ValueError('bounded presentation required')
        old=prior/'native'/row['name']
        if (old/'manifest.json').is_file():
            m=json.loads((old/'manifest.json').read_text());_validate_manifest(m)
            if m['source']!=req.model_dump() or not m.get('hidden_capture_native_outputs_bitwise_equal'):
                raise ValueError('completed prior capture is not authentic')
            pins={'fmri.npz':m['array_sha256'],**{r['file']:r['sha256'] for r in m['dense_features']+m['hidden_features']}}
            for name,digest in pins.items():
                if Path(name).name!=name or file_sha256(old/name)!=digest:raise ValueError('prior native arrays changed')
            dest=native/row['name'];dest.mkdir();shutil.copyfile(old/'manifest.json',dest/'manifest.json')
            for name in pins:shutil.copyfile(old/name,dest/name)
            reused.append(row['name'])
        else:
            if row['name']=='text':raise ValueError('committed text capture must be reused, never re-extracted')
            if time.monotonic()-started>940:raise TimeoutError('remaining native/replay budget exhausted')
            run_tribe(media,req,native/row['name'],Path('/model-cache'));computed.append(row['name'])
            gc.collect();torch.cuda.empty_cache()
        cache.commit()
        print(json.dumps({'native_complete':row['name'],'reused':row['name'] in reused,'elapsed_seconds':time.monotonic()-started}),flush=True)
    if time.monotonic()-started>1080:raise TimeoutError('insufficient remaining audio replay budget')
    request=prepare_request({'image':native/'image','pdf':native/'pdf'},native/'text')
    replay=root/'audio-replay';receipt=run_replay(request,replay,Path('/model-cache'))
    for record in request['sources']:
        for r in record['dense']:(replay/record['name']/r['file']).write_bytes(r['blob'])
    proof={'schema':'perception-narrated-native-batch-v1','complete':True,'elapsed_seconds':time.monotonic()-started,
        'sources':{r['name']:r['request']['sha256'] for r in rows},'replay_qualified':receipt['passed'],
        'cloud_result_cache_key':batch_key,'participant_data_received':False,'EEG_weights_received':False,'paid_retries':0,
        'explicit_recovery_of_timeout':True,'prior_interrupted_outputs_preserved':True,'reused_native_sources':reused,
        'computed_native_sources':computed,'native_and_EEG_audio_features_share_presented_media':True,
        'native_manifests':{r['name']:file_sha256(native/r['name']/'manifest.json') for r in rows}}
    (root/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n');cache.commit()
    files=[p for p in root.rglob('*') if p.is_file()]
    if sum(p.stat().st_size for p in files)>256*1024**2:raise ValueError('output bound exceeded')
    archives=Path('/model-cache/narrated-numerical-recovery-archives');archives.mkdir(exist_ok=True);archive=archives/(batch_key+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for path in files:z.write(path,str(path.relative_to(root)))
    cache.commit()
    return {'archive_path':str(archive.relative_to('/model-cache')),'archive_sha256':file_sha256(archive),'archive_bytes':archive.stat().st_size,'report':proof}
