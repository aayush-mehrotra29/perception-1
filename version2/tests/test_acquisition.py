import io
import zipfile
import pytest
from speech_eeg import acquire_alice


def test_range_member_acquisition_checks_header_and_crc(tmp_path,monkeypatch):
    data=b'known EEG-format fixture bytes'*100
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('eeg/S03/S03_alice-raw.fif',data)
    content=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as z:info=z.infolist()[0]
    archive={'URL':'https://example.org/archive','archive_bytes':len(content),'provider_archive_MD5_not_independently_whole_verified':'dummy'}
    entry={'name':info.filename,'local_header_offset':info.header_offset,'compression_method':8,
        'compressed_size_bytes':info.compress_size,'uncompressed_size_bytes':info.file_size,'CRC32':f'{info.CRC:08x}'}
    monkeypatch.setattr(acquire_alice,'request_range',lambda url,start,stop,total,etag:io.BytesIO(content[start:stop+1]))
    out=tmp_path/'S03.fif';r=acquire_alice.acquire_member(archive,entry,out,lambda:None)
    assert out.read_bytes()==data and r['CRC32']==entry['CRC32']
    entry['CRC32']='00000000'
    with pytest.raises(ValueError,match='header disagrees'):
        acquire_alice.acquire_member(archive,entry,tmp_path/'invalid.fif',lambda:None)
