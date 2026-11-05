import json

import numpy as np
import pytest

from brain.worker_contract import file_sha256
from lesson_harness_v1.tests.test_app_release import fixture
from numerical_brain_v1 import export as module


def dump(path,value):path.write_text(json.dumps(value))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'guard',lambda _:None)
    root=tmp_path/'scores';fixture(root,all_heads=True)
    native=tmp_path/'native';native.mkdir();cortex=np.arange(50*20484,dtype=np.float32).reshape(50,20484)
    np.savez(native/'fmri.npz',predicted_cortical_response=cortex)
    media=tmp_path/'media.mp4';media.write_bytes(b'authorized synthetic media fixture')
    shared=file_sha256(media)
    nm={'schema':'perception-native-tribe-v1','source':{'sha256':shared},'array_sha256':file_sha256(native/'fmri.npz'),
        'shape':list(cortex.shape),'neuro_offset_seconds':5.,
        'segments':[{'start':i,'duration':1.,'timeline':'fixture'} for i in range(50)]}
    dump(native/'manifest.json',nm)
    eeg=tmp_path/'eeg';eeg.mkdir();times=np.arange(2251)/50;supported=(times>=2)&(times<=43)
    values=np.full((len(times),40),np.nan,dtype=np.float32);values[supported]=1e-6
    np.savez(eeg/'eeg.npz',predicted_eeg_volts=values,query_seconds=times,supported=supported)
    em={'schema':'perception-numerical-waveform-v1','complete':True,'source_sha256':shared,'unit':'V','rate_hz':50,
        'filter_hz':[.5,8.],'channels':list(range(40)),'target':'filtered average-listener EEG',
        'total_samples':len(times),'source_offset_seconds':0.,'duration_seconds':45.,
        'supported_samples':int(supported.sum()),'files':{'eeg.npz':file_sha256(eeg/'eeg.npz')}}
    dump(eeg/'manifest.json',em)
    tm=json.loads((root/'timeline.json').read_text());tm['source_sha256']=shared;dump(root/'timeline.json',tm)
    context={'schema':'perception-numerical-neural-context-v1','points':[]}
    for second in range(46):
        supported_point=second>=44
        context['points'].append({'second':second,'quality_prediction_supported':supported_point,
            'available':{'predicted_eeg':False,'predicted_cortex':supported_point},
            'used_by_quality_prediction':{'predicted_eeg':False,'predicted_cortex':supported_point}})
    dump(root/'neural-context.json',context)
    sm=json.loads((root/'manifest.json').read_text());sm.update(native_manifest_sha256=file_sha256(native/'manifest.json'),
        eeg_manifest_sha256=file_sha256(eeg/'manifest.json'),neural_context={'file':'neural-context.json',
        'sha256':file_sha256(root/'neural-context.json'),'used_points':{'predicted_eeg':0,'predicted_cortex':2}})
    sm['quality_timeline']['sha256']=file_sha256(root/'timeline.json');dump(root/'manifest.json',sm)
    surface=tmp_path/'surface';surface.mkdir();arrays={'pial_left_coordinates':np.zeros((3,3),np.float32),
        'pial_left_faces':np.array([[0,1,2]],np.int32)}
    np.savez(surface/'surface.npz',**arrays);dump(surface/'manifest.json',{'test':True})
    monkeypatch.setattr(module,'load_surface',lambda _:({'geometry_role':'synthetic test mesh'},arrays))
    presentation=tmp_path/'presentation';presentation.mkdir();dump(presentation/'manifest.json',{
        'schema':'numerical-narrated-presentation-v1','complete':True,'parent':{'sha256':shared,'media_kind':'video'},
        'derivative':{'sha256':shared,'media_kind':'video'},'media_path':str(media),'both_branches_require_derivative_sha256':shared,
        'mode':'native_passthrough','duration_seconds':45.,'original_clock_preserved':True,'narration_generated':False})
    return eeg,native,root,surface,presentation,values


def test_buffers_native_tail_and_per_query_neural_masks(tmp_path,monkeypatch):
    eeg,native,scores,surface,presentation,values=setup(tmp_path,monkeypatch)
    output=tmp_path/'export';result=module.export(eeg,native,scores,surface,output,narration_directory=presentation)
    for name,source in [('fmri.npz',native),('eeg.npz',eeg),('neural-context.json',scores)]:
        assert file_sha256(output/name)==file_sha256(source/name)
    restored=np.fromfile(output/'eeg-volts.bin',dtype='<f4').reshape(values.shape)
    assert restored.tobytes()==values.tobytes()
    assert result['fmri']['neuro_offset_seconds']==5 and result['fmri']['post_presentation_tail_seconds']==5
    assert result['scores']['neural_context']['used_points']=={'predicted_eeg':0,'predicted_cortex':2}
    assert result['eeg']['unit']=='V' and not result['fmri']['observed_scanner_data']


@pytest.mark.parametrize('bad',['clock','missing_zero','wrong_source','context_usage','narration_status'])
def test_reject_mismatched_or_invented_numerical_streams(tmp_path,monkeypatch,bad):
    eeg,native,scores,surface,presentation,_=setup(tmp_path,monkeypatch)
    em=json.loads((eeg/'manifest.json').read_text())
    if bad in {'clock','missing_zero'}:
        with np.load(eeg/'eeg.npz') as z:arrays={k:z[k] for k in z.files}
        if bad=='clock':arrays['query_seconds'][1]=.04
        else:arrays['predicted_eeg_volts'][0]=0
        np.savez(eeg/'eeg.npz',**arrays);em['files']['eeg.npz']=file_sha256(eeg/'eeg.npz')
    elif bad=='wrong_source':em['source_sha256']='b'*64
    elif bad=='context_usage':
        sm=json.loads((scores/'manifest.json').read_text());sm['neural_context']['used_points']['predicted_eeg']=1
        dump(scores/'manifest.json',sm)
    else:
        pm=json.loads((presentation/'manifest.json').read_text());pm['narration_generated']=True
        pm['parent']['media_kind']='image';pm['original_clock_preserved']=False;pm['mode']='narrated_simulation'
        dump(presentation/'manifest.json',pm)
    dump(eeg/'manifest.json',em)
    sm=json.loads((scores/'manifest.json').read_text());sm['eeg_manifest_sha256']=file_sha256(eeg/'manifest.json');dump(scores/'manifest.json',sm)
    with pytest.raises(ValueError):module.export(eeg,native,scores,surface,tmp_path/'export',narration_directory=presentation)
    assert not (tmp_path/'export').exists()
