import io
import zipfile
import pytest
from speech_eeg.bbbd_inventory import RangeFile, summarize


def test_bounded_zip_inventory_and_metadata_read():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('README.md','fixture')
        z.writestr('sub-01/ses-01/eeg/sub-01_ses-01_task-stim01_eeg.bdf',b'not-decoded')
    source=b.getvalue()
    def fetch(start,stop):
        return 206,{'Content-Range':f'bytes {start}-{stop}/{len(source)}','ETag':'fixture'},source[start:stop+1]
    r=RangeFile('https://example.org/archive',len(source),fetch)
    with zipfile.ZipFile(r) as z:
        assert z.read('README.md')==b'fixture'
        s=summarize(z.infolist());assert s['root_EEG_recordings']==1
        assert s['root_EEG_by_task']=={'stim01':1} and not s['EEG_quality_assessed']
    r.verify_bookends()


def test_ignored_or_changing_ranges_and_oversized_reads_rejected():
    r=RangeFile('url',10,lambda a,b:(200,{},b'0123456789'))
    with pytest.raises(ValueError,match='range'):r.read(2)
    r=RangeFile('url',10,lambda a,b:(206,{'Content-Range':f'bytes {a}-{b}/10','ETag':str(a)},b'x'*(b-a+1)))
    r.read(1)
    with pytest.raises(ValueError,match='changed'):r.read(1)
    r=RangeFile('url',100,budget=1)
    with pytest.raises(ValueError,match='budget'):r.read(2)


def test_metadata_bookends_without_http_validator_detect_mutation():
    body=bytearray(b'0123456789')
    def fetch(a,b):return 206,{'Content-Range':f'bytes {a}-{b}/10'},bytes(body[a:b+1])
    r=RangeFile('url',10,fetch);r.read(4);r.verify_bookends()
    body[0]=ord('x')
    with pytest.raises(ValueError,match='bookends'):r.verify_bookends()
