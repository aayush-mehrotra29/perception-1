"""Pinned, resumable local teacher acquisition; no participant data is transmitted."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

from brain.credentials import load_credentials
from brain.edugage_targets import guard
from brain.worker_contract import file_sha256

REPO='meta-llama/Llama-3.2-3B-Instruct'
REVISION='0cb88a4f764b7a12671c53f0838cd831a0843b95'
FILES=('LICENSE.txt','USE_POLICY.md','config.json','generation_config.json',
       'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors',
       'model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json')


def matches(path,record):
    if not path.is_file() or path.stat().st_size!=record['size']:return False
    if record.get('lfs'):return file_sha256(path)==record['lfs']['sha256']
    digest=hashlib.sha1(f'blob {record["size"]}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()==record['blobId']


def acquire(output,credentials):
    output=Path(output);guard(output);load_credentials(Path(credentials))
    token=os.environ.get('HF_TOKEN')
    if not token:raise RuntimeError('HF_TOKEN required locally')
    headers={'Authorization':'Bearer '+token,'User-Agent':'Perception-local-checkpoint/1'}
    url=f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as response:
        meta=json.load(response)
    if meta['sha']!=REVISION:raise ValueError('teacher revision mismatch')
    records={r['rfilename']:r for r in meta['siblings'] if r['rfilename'] in FILES}
    if set(records)!=set(FILES):raise ValueError('required checkpoint file missing')
    output.mkdir(parents=True,exist_ok=True)
    remaining=sum(r['size']-(output/(name+'.partial')).stat().st_size
                  if (output/(name+'.partial')).exists() else r['size']
                  for name,r in records.items() if not matches(output/name,r))
    if shutil.disk_usage(output).free-remaining<12*1024**3:
        raise RuntimeError('checkpoint would cross external 12 GiB reserve')
    receipt={'schema':'perception-local-teacher-acquisition-v1','repo':REPO,'revision':REVISION,
             'participant_data_transmitted':False,'complete':False,'files':{}}
    for name in FILES:
        guard(output);record=records[name];path=output/name
        if not matches(path,record):
            if path.exists():raise ValueError('existing checkpoint file failed identity check: '+name)
            partial=output/(name+'.partial');offset=partial.stat().st_size if partial.exists() else 0
            request_headers=dict(headers)
            if offset:request_headers['Range']=f'bytes={offset}-'
            request=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',headers=request_headers)
            with urllib.request.urlopen(request,timeout=60) as response:
                if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise RuntimeError('server did not honor checkpoint resume range')
                with partial.open('ab' if offset else 'wb') as stream:
                    while chunk:=response.read(8*1024**2):
                        guard(output);stream.write(chunk)
                        if stream.tell()>record['size']:raise ValueError('checkpoint exceeds pinned size')
            if not matches(partial,record):raise ValueError('checkpoint content hash mismatch: '+name)
            partial.rename(path)
        receipt['files'][name]={'size':record['size'],'sha256':file_sha256(path),'upstream_blob':record['blobId']}
        (output/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('verified '+name,flush=True)
    receipt['complete']=True
    (output/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--credentials',type=Path,required=True)
    a=p.parse_args();acquire(a.output,a.credentials)
