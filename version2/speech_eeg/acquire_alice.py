"""Bounded, checksum-checked ZIP member acquisition from the author archive.

Downloads only named source FIF members; no EEG decoding, fitting or scoring.
"""
import argparse
import struct
import time
import urllib.request
import zlib
from pathlib import Path
from .common import read,write,sha,guard


def request_range(url,start,stop,total,etag):
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{stop}',
        'If-Range':'"'+etag+'"','Accept-Encoding':'identity','User-Agent':'Perception-Research/2'})
    response=urllib.request.urlopen(request,timeout=45)
    if (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop}/{total}'
        or response.headers.get('ETag','').strip('"')!=etag):
        response.close();raise ValueError('Archive identity or byte-range response changed')
    return response


def acquire_member(archive,entry,destination,check):
    offset=entry['local_header_offset'];url=archive['URL'];total=archive['archive_bytes']
    etag=archive['provider_archive_MD5_not_independently_whole_verified']
    with request_range(url,offset,offset+29,total,etag) as response:header=response.read(31)
    if len(header)!=30 or header[:4]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header')
    fields=struct.unpack('<4s5H3I2H',header)
    _,_,flags,method,_,_,crc,compressed,size,name_length,extra_length=fields
    if (flags!=0 or method!=8 or compressed!=entry['compressed_size_bytes'] or size!=entry['uncompressed_size_bytes']
        or f'{crc:08x}'!=entry['CRC32'] or name_length+extra_length>65536):raise ValueError('ZIP header disagrees with archived central directory')
    with request_range(url,offset+30,offset+29+name_length+extra_length,total,etag) as response:
        variable=response.read(65537)
    if variable[:name_length].decode('utf8')!=entry['name'] or len(variable)!=name_length+extra_length:
        raise ValueError('ZIP member name or extent changed')
    start=offset+30+name_length+extra_length;stop=start+compressed-1
    temp=destination.with_suffix(destination.suffix+'.partial');decompress=zlib.decompressobj(-15)
    got=written=observed_crc=0
    with request_range(url,start,stop,total,etag) as response,temp.open('xb') as target:
        while block:=response.read(1024**2):
            check();got+=len(block)
            if got>compressed:raise ValueError('Excess compressed bytes')
            raw=decompress.decompress(block,2*1024**2)
            if decompress.unconsumed_tail:raise ValueError('Excessive decompression expansion')
            written+=len(raw);observed_crc=zlib.crc32(raw,observed_crc);target.write(raw)
            if written>size:raise ValueError('Excess uncompressed bytes')
    if got!=compressed or written!=size or not decompress.eof or decompress.unused_data or observed_crc!=crc:
        raise ValueError('Incomplete or corrupted source member; partial retained')
    temp.rename(destination)
    return {'file':destination.name,'sha256':sha(destination),'size_bytes':written,
            'compressed_bytes_received':got,'CRC32':f'{observed_crc:08x}','source_member':entry['name']}


def run(config_path,output):
    c=read(config_path);resource=read(c['resource_config']);guard(resource)
    metadata=Path(c['archive_directory'])
    if sha(metadata)!=c['archive_directory_sha256']:raise ValueError('Pinned archive directory changed')
    archive=next(a for a in read(metadata)['archives'] if a['name']=='eeg.0.zip')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write(output/'config.json',c)
    deadline=time.monotonic()+1800;records=[]
    def check():
        guard(resource)
        if time.monotonic()>deadline:raise RuntimeError('Acquisition runtime limit reached; no retry')
    for person in c['participants']:
        check();name=f'eeg/{person}/{person}_alice-raw.fif'
        entry=next(e for e in archive['entries'] if e['name']==name)
        record=acquire_member(archive,entry,output/f'{person}_alice-raw.fif',check)
        records.append({'participant':person,**record});write(output/'progress.json',records)
        print(person+' source FIF verified',flush=True)
    write(output/'manifest.json',{'schema':'perception-v2-alice-cohort-source','complete':True,'files':records,
        'source_url':archive['URL'],'archive_ETag':archive['provider_archive_MD5_not_independently_whole_verified'],
        'archive_MD5_independently_recomputed':False,'member_CRC32_verified':True,
        'config_sha256':sha(config_path),'code_sha256':sha(__file__),'EEG_samples_decoded':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');a=p.parse_args();run(a.config,a.output)
