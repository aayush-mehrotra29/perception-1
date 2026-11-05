import copy
import json

import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1.app_release import export as browser_export
from lesson_harness_v1.combined_export import TARGETS,export,validate_eeg
from lesson_harness_v1.combined_preview import build
from lesson_harness_v1.tests.test_app_release import fixture


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr('lesson_harness_v1.app_release.guard',lambda _:None)
    monkeypatch.setattr('lesson_harness_v1.combined_export.guard',lambda _:None)
    monkeypatch.setattr('lesson_harness_v1.combined_preview.guard',lambda _:None)
    source=tmp_path/'source';fixture(source,all_heads=True)
    manifest=json.loads((source/'manifest.json').read_text());manifest['feature_grid_manifest_sha256']='c'*64
    (source/'manifest.json').write_text(json.dumps(manifest))
    browser=tmp_path/'browser';browser_export(source,browser,require_all_heads=True)
    response=json.loads((browser/'app-response.json').read_text())
    points=[]
    for p in response['timeline']:
        valid=p['second']>=44
        points.append({'second':p['second'],'context_start':p['context_start'],'context_stop':p['context_stop'],
            'active_modalities':p['active_modalities'],
            'values':dict(zip(TARGETS,[-50.,-20.,4.,5.] if valid else [None]*4)),
            'ensemble_sd':dict.fromkeys(TARGETS,.1 if valid else None),
            'status':'experimental_prediction' if valid else 'unavailable','reason':None if valid else 'incomplete_context'})
    eeg={'schema':'perception-eeg-summary-timeline-v1','source_sha256':'a'*64,'artifact_sha256':'b'*64,
        'input_format':'video','duration_seconds':45.,'clock_basis':'media',
        'context_seconds':44,'query_step_seconds':1,'target_order':list(TARGETS),
        'units':{'mean':'microvolts','log1p_variance':'log1p(microvolts squared)'},'points':points}
    eegroot=tmp_path/'eeg';eegroot.mkdir();(eegroot/'eeg-timeline.json').write_text(json.dumps(eeg))
    eeg_manifest={'schema':'perception-eeg-summary-bundle-v1','complete':True,'source_sha256':'a'*64,
        'artifact_sha256':'b'*64,'files':{'eeg-timeline.json':file_sha256(eegroot/'eeg-timeline.json')},
        'native_manifest_sha256':file_sha256(browser/'native-tribe-manifest.json'),
        'feature_grid_manifest_sha256':'c'*64,'supported_points':2,'unsupported_points':44,'TRIBE_arrays_modified':False}
    (eegroot/'manifest.json').write_text(json.dumps(eeg_manifest))
    return browser,eegroot,response,eeg


def test_combined_preserves_all_arrays_and_does_not_claim_waveforms(tmp_path,monkeypatch):
    browser,eegroot,response,eeg=setup(tmp_path,monkeypatch)
    report=export(browser,eegroot,tmp_path/'combined')
    assert report['supported_EEG_points']==2
    for name in ('timeline.json','fmri.npz','cortical-response.bin','native-tribe-manifest.json'):
        assert file_sha256(browser/name)==file_sha256(tmp_path/'combined'/name)
    result=json.loads((tmp_path/'combined/combined-response.json').read_text())
    assert result['eeg']==eeg
    assert not result['eeg_waveform_generated']
    assert not result['qualities_consume_predicted_eeg_values']
    assert result['quality_response']==response
    built=build(tmp_path/'combined/combined-response.json',tmp_path/'playback.html')
    assert not built['raw_waveform_synthesis']
    html=(tmp_path/'playback.html').read_text()
    assert 'requestAnimationFrame(playbackFrame)' in html
    assert 'Predicted EEG summaries' in html and 'precomputed streams' in html
    assert 'frequency bands, or measured student activity' in html


@pytest.mark.parametrize('kind',['artifact','clock','grid','warmup','variance','null'])
def test_reject_mismatched_or_invalid_eeg(tmp_path,monkeypatch,kind):
    _,_,response,eeg=setup(tmp_path,monkeypatch)
    bad=copy.deepcopy(eeg)
    if kind=='artifact':bad['artifact_sha256']='d'*64
    elif kind=='clock':bad['clock_basis']='invented'
    elif kind=='grid':bad['points'][44]['second']=45
    elif kind=='warmup':bad['points'][0]=copy.deepcopy(bad['points'][44]);bad['points'][0]['second']=0
    elif kind=='variance':bad['points'][44]['values']['log1p_variance_AF7']=-1
    else:bad['points'][0]['values']['mean_AF7']=0
    with pytest.raises(ValueError):validate_eeg(bad,response)
