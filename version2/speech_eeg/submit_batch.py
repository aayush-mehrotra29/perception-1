"""One persisted batch dispatch under the original shared project credit cap."""
import io
import json
import hashlib
import zipfile
from pathlib import Path
from .common import read,write,sha
from brain.credentials import load_credentials
from brain.credits import CreditLedger
from brain.modal_client import single_gpu_job,pending_dispatches,unpack_result
from brain.worker_contract import PublicMediaRequest,validate_media
from brain.edugage_targets import guard
from brain.observe_call import observe


def run():
    import modal
    root=Path(__file__).resolve().parents[1];state=root.parent/'.local'
    load_credentials(state/'brain-credentials.json')
    design=read(root/'configs/tribe_batch_requests.json');output=root/'runs/tribe-hidden-batch'
    identifier='v2-alice02-12-hidden-batch-20261103';guard(output)
    if output.exists():raise ValueError('Inspect existing batch; no implicit retry')
    items=[]
    for item in design['items']:
        p=Path(item['local_media']);request=PublicMediaRequest.model_validate(item['request']);validate_media(p,request)
        items.append({'name':item['name'],'request':request.model_dump(),'media':p.read_bytes()})
    with single_gpu_job(state/'brain-gpu.lock'):
        if pending_dispatches(state):raise RuntimeError('Unresolved earlier paid dispatch')
        CreditLedger(state/'brain-credits.sqlite3').reserve(identifier,'modal',4000000)
        call=modal.Function.from_name('perception-brain-v2','tribe_batch').spawn(items)
        dispatch=state/'brain-dispatches'/(hashlib.sha256(identifier.encode()).hexdigest()+'.json')
        with dispatch.open('x') as f:
            json.dump({'schema':'perception-modal-dispatch-v1','request_id':identifier,
                'function_call_id':call.object_id,'modal_app':'perception-brain-v2','kind':'tribe_batch',
                'output':str(output),'reserved_microdollars':4000000,'batch_design':str(root/'configs/tribe_batch_requests.json'),
                'batch_design_sha256':sha(root/'configs/tribe_batch_requests.json')},f,indent=2)
        payload=observe(call,output,identifier);guard(output)
        if len(payload)>256*1024**2:raise ValueError('Excessive batch payload')
        output.mkdir(parents=True,exist_ok=False)
        # Preserve remote result before publishing; a local validation failure must not trigger another paid call.
        (output/'received.zip').write_bytes(payload)
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            entries=z.infolist();names=[e.filename for e in entries]
            if len(names)!=len(set(names)) or sum(e.file_size for e in entries)>256*1024**2:
                raise ValueError('Invalid batch archive sizes/duplicates')
            prefixes={i['name'] for i in items}
            if any(len(Path(n).parts)!=2 or Path(n).parts[0] not in prefixes or '\\' in n or '..' in Path(n).parts for n in names):
                raise ValueError('Invalid batch archive paths')
            for item in items:
                buffer=io.BytesIO()
                with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as single:
                    for entry in entries:
                        if entry.filename.startswith(item['name']+'/'):
                            single.writestr(Path(entry.filename).name,z.read(entry))
                unpack_result(buffer.getvalue(),output/item['name'],'tribe',PublicMediaRequest.model_validate(item['request']),publication_guard=guard,require_hidden=True)
        write(output/'request-receipt.json',{'request_id':identifier,'reserved_microdollars':4000000,
            'billing_reconciled':False,'received_archive_sha256':sha(output/'received.zip'),
            'sources':[i['name'] for i in items],'EEG_or_local_weights_sent':False})
        print('Native hidden batch received and authenticated',flush=True)


if __name__=='__main__':run()
