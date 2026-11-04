import hashlib
import io
import json
import zipfile
import pytest
from brain.modal_client import unpack_result
from brain.worker_contract import PublicMediaRequest


def test_hidden_transport_requires_verified_hashes_and_rejects_extra_files(tmp_path):
    request=PublicMediaRequest(source_id='fixture',sha256='1'*64,source_url='https://example.org/stimulus',
        cloud_use_basis='Public generated test fixture; no neural data.',media_kind='audio',size_bytes=3)
    files={'fmri.npz':b'cortex','dense-0000.npz':b'content','hidden-0000.npz':b'hidden'}
    digest=lambda name:hashlib.sha256(files[name]).hexdigest()
    m={'schema':'perception-native-tribe-v1','source':request.model_dump(),'presentation':None,
       'array_sha256':digest('fmri.npz'),'dense_features':[{'file':'dense-0000.npz','sha256':digest('dense-0000.npz')}],
       'hidden_features':[{'file':'hidden-0000.npz','sha256':digest('hidden-0000.npz')}],
       'hidden_capture_native_outputs_bitwise_equal':True}
    def archive(extra=None):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            z.writestr('manifest.json',json.dumps(m))
            for n,v in {**files,**(extra or {})}.items():z.writestr(n,v)
        return b.getvalue()
    unpack_result(archive(),tmp_path/'valid','tribe',request,require_hidden=True)
    with pytest.raises(ValueError,match='unexpected or missing'):
        unpack_result(archive({'participant-eeg.npz':b'forbidden'}),tmp_path/'invalid','tribe',request)
    m['hidden_features'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='checksum'):
        unpack_result(archive(),tmp_path/'corrupt','tribe',request)
    assert not (tmp_path/'invalid').exists() and not (tmp_path/'corrupt').exists()
    m['hidden_features']=[]
    with pytest.raises(ValueError,match='hidden features required'):
        unpack_result(archive(),tmp_path/'missing','tribe',request,require_hidden=True)
