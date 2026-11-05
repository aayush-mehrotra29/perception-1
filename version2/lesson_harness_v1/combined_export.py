"""Combine immutable quality, EEG-summary, and native TRIBE output streams."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .app_release import checked_child, verify_bundle

TARGETS=('mean_AF7','mean_AF8','log1p_variance_AF7','log1p_variance_AF8')


def validate_eeg(timeline,response):
    if (timeline.get('schema')!='perception-eeg-summary-timeline-v1'
            or timeline.get('source_sha256')!=response['source_sha256']
            or not re.fullmatch('[0-9a-f]{64}',timeline.get('artifact_sha256',''))
            or timeline.get('artifact_sha256')!=response['artifact_sha256']
            or timeline.get('input_format')!=response['input_format']
            or timeline.get('duration_seconds')!=response['duration_seconds']
            or timeline.get('clock_basis')!=response['clock_basis']
            or timeline.get('target_order')!=list(TARGETS)
            or timeline.get('context_seconds')!=44 or timeline.get('query_step_seconds')!=1):
        raise ValueError('EEG identity, target order or context differs')
    if [p['second'] for p in timeline['points']]!=[p['second'] for p in response['timeline']]:
        raise ValueError('EEG and quality query grids differ')
    for point in timeline['points']:
        if set(point['values'])!=set(TARGETS) or set(point['ensemble_sd'])!=set(TARGETS):
            raise ValueError('four explicit EEG summaries required')
        if point['status']=='unavailable':
            if (not point.get('reason') or any(v is not None for v in point['values'].values())
                    or any(v is not None for v in point['ensemble_sd'].values())):
                raise ValueError('unavailable EEG summaries must be null with a reason')
        elif point['status'] in {'experimental_prediction','experimental_transfer'}:
            if (point['second']<44 or point.get('context_start')!=point['second']-44
                    or point.get('context_stop')!=point['second']):
                raise ValueError('EEG prediction requires complete 44-second context')
            for name,value in point['values'].items():
                sd=point['ensemble_sd'][name]
                if (type(value) not in (int,float) or not math.isfinite(value)
                        or type(sd) not in (int,float) or not math.isfinite(sd) or sd<0
                        or ('variance' in name and value<0)):
                    raise ValueError('invalid supported EEG numeric summary')
        else:
            raise ValueError('unknown EEG status')
    return timeline


def export(browser_source,eeg_source,output):
    browser_source,eeg_source,output=map(Path,(browser_source,eeg_source,output))
    guard(output)
    if output.exists():
        raise ValueError('preserve existing combined output')
    browser_manifest,_,_,_,_=verify_bundle(browser_source,require_all_heads=True)
    receipt=json.loads((browser_source/'release-verification.json').read_text())
    if not receipt.get('complete') or not receipt.get('all_five_heads_supported'):
        raise ValueError('complete five-head browser release required')
    for name,digest in receipt['files'].items():
        checked_child(browser_source,name,digest)
    response=json.loads((browser_source/'app-response.json').read_text())
    checked_child(browser_source,response['fmri']['file'],response['fmri']['sha256'])
    eeg_manifest=json.loads((eeg_source/'manifest.json').read_text())
    if (eeg_manifest.get('schema')!='perception-eeg-summary-bundle-v1'
            or eeg_manifest.get('complete') is not True
            or eeg_manifest.get('TRIBE_arrays_modified') is not False
            or eeg_manifest.get('native_manifest_sha256')!=browser_manifest['fmri']['native_manifest_sha256']
            or eeg_manifest.get('feature_grid_manifest_sha256')!=browser_manifest['feature_grid_manifest_sha256']):
        raise ValueError('qualified EEG-summary bundle required')
    member={'file':'eeg-timeline.json','sha256':eeg_manifest['files']['eeg-timeline.json']}
    eeg_path=checked_child(eeg_source,member['file'],member['sha256'])
    eeg=validate_eeg(json.loads(eeg_path.read_text()),response)
    if (eeg_manifest.get('source_sha256')!=response['source_sha256']
            or eeg_manifest.get('artifact_sha256')!=eeg['artifact_sha256']):
        raise ValueError('EEG manifest and timeline identity differ')
    supported=sum(p['status']!='unavailable' for p in eeg['points'])
    if (supported<1 or eeg_manifest.get('supported_points')!=supported
            or eeg_manifest.get('unsupported_points')!=len(eeg['points'])-supported):
        raise ValueError('three-stream release requires supported EEG summaries and matching coverage counts')
    output.mkdir(parents=True,exist_ok=False)
    for name,digest in receipt['files'].items():
        shutil.copyfile(browser_source/name,output/name)
        if file_sha256(output/name)!=digest:
            raise ValueError('browser member changed during copy')
    shutil.copyfile(browser_source/'release-verification.json',output/'quality-release-verification.json')
    if (browser_source/'SUPERVISION.md').exists():
        shutil.copyfile(browser_source/'SUPERVISION.md',output/'SUPERVISION.md')
    shutil.copyfile(eeg_path,output/'eeg-timeline.json')
    shutil.copyfile(eeg_source/'manifest.json',output/'eeg-manifest.json')
    if file_sha256(output/'eeg-timeline.json')!=member['sha256']:
        raise ValueError('EEG timeline changed during copy')
    combined={'schema':'perception-three-stream-response-v1',
        'source_sha256':response['source_sha256'],'input_format':response['input_format'],
        'duration_seconds':response['duration_seconds'],'quality_response':response,'eeg':eeg,
        'analysis_mode':'whole_lesson_offline','playback_mode':'precomputed_synchronized_streams',
        'inference_at_playback_time':False,'causal_realtime_prediction_validated':False,
        'eeg_waveform_generated':False,'recorded_student_eeg_included':False,
        'qualities_consume_predicted_eeg_values':False,
        'architecture_note':'Quality prediction uses an EEG-supervised shared representation and content/TRIBE features. The explicit EEG-summary readout is a parallel output; its predicted numeric values are not inputs to the existing quality heads.',
        'native_TRIBE_modified':False,
        'display_note':'EEG means and log1p variances summarize 44-second contexts; lines between display queries do not constitute raw electrical waveforms.'}
    (output/'combined-response.json').write_text(json.dumps(combined,indent=2,allow_nan=False)+'\n')
    report={'schema':'perception-three-stream-export-check-v1','complete':True,
        'all_five_quality_heads_supported':True,
        'supported_EEG_points':sum(p['status']!='unavailable' for p in eeg['points']),
        'identical_quality_eeg_integer_grids':True,'native_TRIBE_bytes_unchanged':True,
        'native_TRIBE_clock_unchanged':True,'EEG_raw_waveform_claimed':False,
        'quality_source_artifact':response['artifact_sha256'],'EEG_source_artifact':eeg['artifact_sha256'],
        'source_browser_receipt_sha256':file_sha256(browser_source/'release-verification.json'),
        'source_eeg_manifest_sha256':file_sha256(eeg_source/'manifest.json'),
        'files':{p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output/'combined-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('browser_source','eeg_source','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(export(a.browser_source,a.eeg_source,a.output),indent=2))
