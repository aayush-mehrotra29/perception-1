"""Pinned public MLX teacher acquisition; no participant content leaves this machine."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .acquire_teacher import matches

REPO = 'mlx-community/Qwen3-8B-4bit'
REVISION = '545dc4251c05440727734bcd94334791f6ab0192'
FILES = ('README.md', 'config.json', 'added_tokens.json', 'merges.txt',
         'model.safetensors', 'model.safetensors.index.json',
         'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json')


def acquire(output):
    output = Path(output)
    guard(output)
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise ValueError('revision mismatch')
    records = {r['rfilename']: r for r in metadata['siblings'] if r['rfilename'] in FILES}
    if set(records) != set(FILES):
        raise ValueError('checkpoint manifest incomplete')
    output.mkdir(parents=True, exist_ok=True)
    (output/'upstream.json').write_text(json.dumps(metadata, indent=2)+'\n')
    remaining = sum(r['size'] for n,r in records.items() if not matches(output/n,r))
    if shutil.disk_usage(output).free - remaining < 12*1024**3:
        raise RuntimeError('download would cross external reserve')
    receipt = {'schema':'perception-public-mlx-teacher-v1','repo':REPO,'revision':REVISION,
               'parent_repo':'Qwen/Qwen3-8B','community_quantization':True,
               'authorization':'user approved stronger local open-weight teacher',
               'participant_data_transmitted':False,'complete':False,'files':{}}
    for name in FILES:
        guard(output)
        target = output/name
        record = records[name]
        if not matches(target,record):
            if target.exists():
                raise ValueError('preserve mismatched existing file: '+name)
            partial = output/(name+'.partial')
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range':f'bytes={offset}-'} if offset else {}
            request = urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',headers=headers)
            with urllib.request.urlopen(request,timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise RuntimeError('resume range not honored')
                with partial.open('ab' if offset else 'wb') as stream:
                    milestone = offset//(256*1024**2)
                    while chunk := response.read(4*1024**2):
                        guard(output)
                        stream.write(chunk)
                        if stream.tell()>record['size']:
                            raise ValueError('download exceeds pinned size')
                        current=stream.tell()//(256*1024**2)
                        if current>milestone:
                            print(f'{name}: {stream.tell()/1024**3:.2f} GiB',flush=True)
                            milestone=current
            if not matches(partial,record):
                raise ValueError('upstream content hash mismatch: '+name)
            partial.rename(target)
        receipt['files'][name]=file_sha256(target)
        (output/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('verified '+name,flush=True)
    config=json.loads((output/'config.json').read_text())
    if config['model_type']!='qwen3' or config['quantization']['bits']!=4:
        raise ValueError('unexpected model architecture or quantization')
    receipt['complete']=True
    (output/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    acquire(parser.parse_args().output)
