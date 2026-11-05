"""Sequential quality screening with explicit exclusions and immutable receipts."""
import argparse
import json
from pathlib import Path

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
def screen_configuration(mode):
    if mode=='four_channel':
        from . import eeg_quality as module
    elif mode=='frontal':
        from . import eeg_quality_frontal as module
    else:raise ValueError('unknown fixed screen mode')
    return module.screen,module.POLICY,Path(module.__file__)


def run(metadata_root,original_targets,output,screen_mode='four_channel'):
    metadata_root,original_targets,output=map(Path,(metadata_root,original_targets,output));guard(output)
    source=json.loads((original_targets/'summary.json').read_text())
    if not source['complete'] or source['expected_viewings']!=32:raise ValueError('complete original corpus audit required')
    screen,policy,code=screen_configuration(screen_mode)
    policy_hash=file_sha256(policy);code_hash=file_sha256(code)
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for entry in source['viewings']:
        guard(output)
        name=f"participant-{entry['participant']:02d}-{entry['lecture']}"
        original=original_targets/Path(entry['report']).parent
        if file_sha256(original/'report.json')!=entry['report_sha256']:raise ValueError('original report identity changed')
        metadata=metadata_root/f"participant-{entry['participant']:02d}"/(entry['lecture']+'.json')
        folder=output/name
        if entry['supported_windows']==0:
            records.append({'participant':entry['participant'],'lecture':entry['lecture'],'status':'excluded',
                'reason':'no_original_supported_windows','original_report_sha256':entry['report_sha256'],'quality_supported':0})
        else:
            try:
                if (folder/'manifest.json').exists():
                    result=json.loads((folder/'manifest.json').read_text())
                    if (not result['complete'] or result['policy_sha256']!=policy_hash or result['code_sha256']!=code_hash
                            or result['metadata_sha256']!=file_sha256(metadata)
                            or result['original_report_sha256']!=entry['report_sha256']):raise ValueError('resume identity mismatch')
                    for filename,digest in result['files'].items():
                        if file_sha256(folder/filename)!=digest:raise ValueError('quality result changed')
                else:result=screen(metadata,original,folder)
                records.append({'participant':entry['participant'],'lecture':entry['lecture'],'status':'screened',
                    'original_supported':result['original_supported'],'quality_supported':result['quality_supported'],
                    'manifest':str((folder/'manifest.json').resolve()),'sha256':file_sha256(folder/'manifest.json')})
            except (ValueError,FileNotFoundError) as error:
                records.append({'participant':entry['participant'],'lecture':entry['lecture'],'status':'excluded',
                    'reason':str(error),'quality_supported':0})
        report={'schema':'perception-received-EEG-quality-corpus-v1','screen_mode':screen_mode,
            'target_channels':['AF7','AF8'] if screen_mode=='frontal' else ['TP9','AF7','AF8','TP10'],
            'complete':len(records)==len(source['viewings']),
            'policy_sha256':policy_hash,'screen_code_sha256':code_hash,'original_summary_sha256':file_sha256(original_targets/'summary.json'),
            'viewings':records,'supported_windows':sum(r['quality_supported'] for r in records),
            'qualification_scope':'fixed engineering screen for exploratory voltage-summary supervision; not physical/neurobehavioral validation',
            'artifact_free_claim':False,'physical_acquisition_timing_established':False}
        temporary=output/'summary.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output/'summary.json')
        print(f"{len(records)}/{len(source['viewings'])} viewings screened; {report['supported_windows']} supported windows",flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('metadata_root',type=Path);p.add_argument('original_targets',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--screen-mode',choices=['four_channel','frontal'],default='four_channel')
    a=p.parse_args();run(a.metadata_root,a.original_targets,a.output,a.screen_mode)
