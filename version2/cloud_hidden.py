"""Bounded content-only TRIBE hidden extraction on the existing model cache."""
import modal

app=modal.App('perception-brain-v2')
cache=modal.Volume.from_name('perception-public-model-cache',create_if_missing=False)
hf=modal.Secret.from_name('perception-huggingface',required_keys=['HF_TOKEN'])
image=(modal.Image.debian_slim(python_version='3.11')
    .apt_install('git','ffmpeg','libsndfile1')
    .pip_install('torch==2.6.0','torchvision==0.21.0','torchaudio==2.6.0','numpy==2.2.6','uv','pydantic>=2,<3')
    .pip_install('git+https://github.com/facebookresearch/tribev2.git@af58661791a351a448a489042a28f6c37e1c14b7')
    .pip_install('transformers==5.17.0','huggingface_hub==1.16.1')
    .env({'OMP_NUM_THREADS':'1'})
    .add_local_python_source('brain','speech_eeg'))


@app.function(image=image,gpu='A100-80GB',cpu=4,memory=65536,timeout=1200,
              retries=0,max_containers=1,min_containers=0,scaledown_window=5,
              volumes={'/model-cache':cache},secrets=[hf])
def tribe(media_bytes:bytes,request_payload:dict):
    import os,io,zipfile,tempfile
    from pathlib import Path
    from brain.worker_contract import PublicMediaRequest,validate_media
    from brain.remote_inference import run_tribe
    os.environ.update(HF_HOME='/model-cache/huggingface',TORCH_HOME='/model-cache/torch',
                      UV_CACHE_DIR='/tmp/perception-uv',UV_LINK_MODE='copy')
    request=PublicMediaRequest.model_validate(request_payload)
    if request.media_kind!='audio' or len(media_bytes)>32*1024**2 or len(media_bytes)!=request.size_bytes:
        raise ValueError('Version2 pilot accepts only bounded authenticated source audio')
    media_root=Path('/model-cache/public-content')/request.sha256;media_root.mkdir(parents=True,exist_ok=True)
    media=media_root/'input.wav'
    if not media.exists():media.write_bytes(media_bytes)
    validate_media(media,request)
    with tempfile.TemporaryDirectory() as scratch:
        output=Path(scratch)/'output';run_tribe(media,request,output,Path('/model-cache'))
        files=list(output.iterdir())
        if not all(p.is_file() for p in files) or sum(p.stat().st_size for p in files)>256*1024**2:
            raise ValueError('Invalid or excessive output')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
            for p in files:z.write(p,p.name)
        cache.commit()
        return buffer.getvalue()


@app.function(image=image,gpu='A100-80GB',cpu=4,memory=65536,timeout=4000,
              retries=0,max_containers=1,min_containers=0,scaledown_window=5,
              volumes={'/model-cache':cache},secrets=[hf])
def tribe_batch(items:list):
    import os,io,zipfile,tempfile,gc,json
    from pathlib import Path
    import torch
    from brain.worker_contract import PublicMediaRequest,validate_media
    from brain.remote_inference import run_tribe
    os.environ.update(HF_HOME='/model-cache/huggingface',TORCH_HOME='/model-cache/torch',
                      UV_CACHE_DIR='/tmp/perception-uv',UV_LINK_MODE='copy')
    if not 1<=len(items)<=11 or sum(len(x['media']) for x in items)>96*1024**2:
        raise ValueError('Batch content bounds exceeded')
    names=[x['name'] for x in items]
    if len(set(names))!=len(names) or any(n not in {f'segment{i:02d}' for i in range(1,13)} for n in names):
        raise ValueError('Invalid batch identifiers')
    with tempfile.TemporaryDirectory() as scratch:
        root=Path(scratch);outputs=[]
        for item in items:
            request=PublicMediaRequest.model_validate(item['request'])
            if request.media_kind!='audio' or len(item['media'])!=request.size_bytes:raise ValueError('Audio only')
            media_root=Path('/model-cache/public-content')/request.sha256;media_root.mkdir(parents=True,exist_ok=True)
            media=media_root/'input.wav'
            if not media.exists():media.write_bytes(item['media'])
            validate_media(media,request)
            output=root/item['name'];run_tribe(media,request,output,Path('/model-cache'));outputs.append(output)
            cache.commit();gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'completed_source':item['name']}),flush=True)
        if sum(p.stat().st_size for o in outputs for p in o.iterdir())>256*1024**2:
            raise ValueError('Batch output transport bound exceeded')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
            for output in outputs:
                for p in output.iterdir():
                    if not p.is_file():raise ValueError('Unexpected output directory')
                    z.write(p,output.name+'/'+p.name)
        return buffer.getvalue()
