"""Bounded official ZIP inventory: acquire metadata, never decode EEG or fit models."""
import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from .common import read, write, sha, guard


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, fetch=None, budget=8*1024**2):
        self.url, self.size, self.position = url, size, 0
        self.fetch = fetch or self._fetch; self.budget = budget
        self.received = 0; self.validators = None; self.requests = 0
        self.read_log = []

    def _fetch(self, start, stop):
        headers = {'Range':f'bytes={start}-{stop}', 'Accept-Encoding':'identity',
                   'User-Agent':'Perception-BBBD-metadata-review/1'}
        if self.validators and self.validators[0]: headers['If-Match'] = self.validators[0]
        with urllib.request.urlopen(urllib.request.Request(self.url,headers=headers),timeout=40) as r:
            # Never consume a full archive when Range is ignored.
            if r.status != 206: raise ValueError('Server ignored bounded range')
            if r.geturl().split('?')[0] != self.url: raise ValueError('Unexpected archive redirect')
            return r.status, dict(r.headers), r.read(stop-start+2)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence==0 else self.position if whence==1 else self.size if whence==2 else -self.size*2)
        if not 0 <= position <= self.size: raise ValueError('Seek outside archive')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size-self.position if size<0 else min(size,self.size-self.position)
        if size==0: return b''
        if size>2*1024**2 or self.received+size>self.budget: raise ValueError('Metadata byte budget exceeded')
        start = self.position; stop = start+size-1
        status, headers, body = self.fetch(start,stop)
        self.received += len(body); self.requests += 1
        h = {k.lower():v for k,v in headers.items()}; validators = (h.get('etag'),h.get('last-modified'))
        if status!=206 or h.get('content-range')!=f'bytes {start}-{stop}/{self.size}' or len(body)!=size:
            raise ValueError('Wrong byte range response')
        if h.get('content-encoding','identity').lower()!='identity': raise ValueError('Encoded range')
        if self.validators is not None and self.validators!=validators: raise ValueError('Archive changed during review')
        self.validators = validators; self.position += size
        self.read_log.append((start,size,hashlib.sha256(body).hexdigest()))
        return body

    def verify_bookends(self):
        # Zenodo currently omits ETag/Last-Modified on these 206 responses.
        # Re-read every used range; this checks only acquired metadata, not the
        # unreceived whole archive. ZIP metadata members additionally pass CRC32.
        for start,size,digest in tuple(self.read_log):
            self.seek(start)
            if hashlib.sha256(self.read(size)).hexdigest()!=digest:
                raise ValueError('Archive metadata changed between bookends')


def summarize(infos):
    raw = [i for i in infos if re.fullmatch(r'sub-[^/]+/ses-[^/]+/eeg/[^/]+_eeg\.bdf',i.filename)]
    derived = [i for i in infos if re.fullmatch(r'derivatives/sub-[^/]+/ses-[^/]+/eeg/[^/]+_desc-eeg\.bdf',i.filename)]
    subjects = sorted({i.filename.split('/')[0] for i in raw})
    by_task = {}
    for i in raw:
        task = re.search(r'task-(stim\d+)',i.filename).group(1)
        by_task[task] = by_task.get(task,0)+1
    return {'entries':len(infos),'subjects_with_root_EEG':subjects,'root_EEG_recordings':len(raw),
            'root_EEG_bytes':sum(i.file_size for i in raw),'root_EEG_by_task':by_task,
            'derived_EEG_recordings':len(derived),'derived_EEG_bytes':sum(i.file_size for i in derived),
            'EEG_quality_assessed':False,'subject_identity_across_experiments_verified':False}


def run(snapshot, output_name='archive-review'):
    if Path(output_name).name!=output_name: raise ValueError('Output must be a single new directory name')
    snapshot = Path(snapshot); output = snapshot/output_name; output.mkdir(exist_ok=False)
    release = read(snapshot/'zenodo-record.json')
    assert release['id']==22968459 and release['metadata']['version']=='1.1'
    results = []
    for experiment in [2,3,4,5]:
        guard(read('configs/stage_a.json'))
        f = next(f for f in release['files'] if f['key']==f'experiment{experiment}.zip')
        url = f['links']['self']; assert url==f'https://zenodo.org/api/records/22968459/files/experiment{experiment}.zip/content'
        folder = output/f'experiment{experiment}'; folder.mkdir()
        stream = RangeFile(url,f['size']); result={'experiment':experiment,'archive_url':url,
                  'archive_size_bytes':f['size'],'provider_archive_checksum':f['checksum'],
                  'whole_archive_checksum_independently_verified':False}
        try:
            with zipfile.ZipFile(stream) as z:
                infos = z.infolist(); result.update(summarize(infos)); metadata = []
                write(folder/'members.json',[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,
                      'CRC32':f'{i.CRC:08x}','offset':i.header_offset} for i in infos])
                # No recording arrays or questionnaire answers are needed for this inventory.
                chosen = [i for i in infos if i.filename in ['README.md','dataset_description.json','participants.json','participants.tsv']
                          or (i.filename.startswith('phenotype/') and i.filename.endswith('.json'))]
                for info in chosen:
                    if info.file_size>512*1024: raise ValueError('Unexpectedly large metadata member')
                    dest=folder/info.filename
                    if not dest.resolve().is_relative_to(folder.resolve()): raise ValueError('Unsafe metadata path')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as handle: handle.write(z.read(info))
                    metadata.append({'file':info.filename,'sha256':sha(dest),'CRC32_verified':True})
                stream.verify_bookends()
                result.update(complete=True,metadata=metadata,all_requested_ranges_replayed_identically=True)
        except Exception as e:
            result.update(complete=False,error_type=type(e).__name__,error=str(e))
        result.update(network_bytes=stream.received,range_requests=stream.requests,http_validators=stream.validators,
                      integrity_scope='Acquired ZIP directory/ranges stable across two reads; metadata member CRC32. Not whole-archive MD5 or EEG integrity.')
        write(folder/'summary.json',result); results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['metadata','subjects_with_root_EEG']}),flush=True)
    write(output/'summary.json',{'source_record':22968459,'metadata_only':True,'EEG_decoded':False,
          'model_fitted':False,'source_sha256':sha(__file__),'experiments':results})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('--output-name',default='archive-review')
    a=p.parse_args();run(a.snapshot,a.output_name)
