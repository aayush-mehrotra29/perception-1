"""One explicit budgeted public-audio call; never retry automatically."""
import argparse
from pathlib import Path
from brain.credentials import load_credentials
from brain.modal_client import submit
from brain.worker_contract import PublicMediaRequest
from .common import read


def run(request_path,output,request_id):
    import modal
    root=Path(__file__).resolve().parents[1];state=root.parent/'.local'
    load_credentials(state/'brain-credentials.json')
    record=read(request_path);request=PublicMediaRequest.model_validate(record['request'])
    with modal.enable_output():
        submit('tribe',Path(record['local_media']),request,Path(output),request_id,
               state=state,reservation_microdollars=2000000,app_name='perception-brain-v2',require_hidden=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('request');p.add_argument('output');p.add_argument('request_id')
    a=p.parse_args();run(a.request,a.output,a.request_id)
