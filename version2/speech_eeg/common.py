import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import plistlib


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def guard(config):
    volume=Path('/Volumes/EEG Research Workspace')
    if not volume.is_mount():raise RuntimeError('Research volume is not mounted')
    info=plistlib.loads(subprocess.check_output(['/usr/sbin/diskutil','info','-plist',str(volume)]))
    if info.get('VolumeUUID')!=config['resources']['research_volume_uuid']:raise RuntimeError('Research volume identity changed')
    for path,key in [('/', 'internal_reserve_bytes'),(volume,'external_reserve_bytes')]:
        if shutil.disk_usage(path).free<config['resources'][key]:raise RuntimeError('Disk reserve reached: '+str(path))


def authenticate(path,record):
    path=Path(path)
    if path.stat().st_size!=record['size_bytes'] or sha(path)!=record['sha256']:
        raise ValueError('Source identity mismatch: '+str(path))
    return {'path':str(path),'sha256':record['sha256'],'size_bytes':record['size_bytes']}
