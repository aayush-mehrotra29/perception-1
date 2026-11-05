"""Backend numerical export with immutable native cortex and explicit EEG masks."""
import json
from pathlib import Path
import shutil

import numpy as np

from brain.edugage_targets import guard
from brain.surface import load_surface
from brain.worker_contract import file_sha256
from lesson_harness_v1.contract import Timeline


def member(root,name,digest):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_sha256(path)!=digest:raise ValueError('source member identity changed')
    return path


def buffer(array,root,name):
    values=np.ascontiguousarray(array,dtype=array.dtype.newbyteorder('<'))
    path=root/name;values.tofile(path)
    reread=np.fromfile(path,dtype=values.dtype).reshape(values.shape)
    if reread.tobytes()!=values.tobytes():raise ValueError('numeric buffer roundtrip changed')
    return {'file':name,'sha256':file_sha256(path),'dtype':values.dtype.str,'shape':list(values.shape),'layout':'C','byte_order':'little'}


def export(eeg_directory,native_directory,scores_directory,surface_directory,output,*,narration_directory,audio_lineage_directory=None):
    eeg_directory,native_directory,scores_directory,surface_directory,output,narration_directory=map(Path,
        (eeg_directory,native_directory,scores_directory,surface_directory,output,narration_directory))
    guard(output)
    if output.exists():raise ValueError('preserve existing numerical bundle')
    presentation=json.loads((narration_directory/'manifest.json').read_text())
    if not presentation.get('complete') or presentation.get('schema')!='numerical-narrated-presentation-v1':raise ValueError('complete shared presentation required')
    shared=presentation['derivative']['sha256']
    if presentation['both_branches_require_derivative_sha256']!=shared or file_sha256(Path(presentation['media_path']))!=shared:raise ValueError('shared presented media changed')
    native=json.loads((native_directory/'manifest.json').read_text())
    if native.get('schema')!='perception-native-tribe-v1':raise ValueError('native cortical manifest required')
    if native['source']['sha256']!=shared:
        if audio_lineage_directory is None:raise ValueError('decoded audio lineage required')
        lineage_directory=Path(audio_lineage_directory);lineage=json.loads((lineage_directory/'manifest.json').read_text())
        if (lineage.get('schema')!='perception-audio-derivative-v1' or lineage['parent']['sha256']!=shared
                or lineage['derivative']!=native['source'] or not lineage['audio_clock_continuity_checked']
                or file_sha256(lineage_directory/'audio.wav')!=native['source']['sha256']):raise ValueError('native and presented audio lineage mismatch')
    native_path=member(native_directory,'fmri.npz',native['array_sha256'])
    with np.load(native_path,allow_pickle=False) as archive: cortex=archive['predicted_cortical_response']
    starts=np.array([s['start'] for s in native['segments']],float);durations=np.array([s['duration'] for s in native['segments']],float)
    if (cortex.shape!=(len(starts),20484) or list(cortex.shape)!=native['shape'] or not np.isfinite(cortex).all()
            or not np.isfinite(starts).all() or not np.isfinite(durations).all() or np.any(durations<=0)
            or np.any(starts[1:]<starts[:-1]+durations[:-1]-1e-8)):raise ValueError('native cortex orclock invalid')
    geometry,arrays=load_surface(surface_directory)
    em=json.loads((eeg_directory/'manifest.json').read_text())
    if (em.get('schema')!='perception-numerical-waveform-v1' or not em.get('complete') or em.get('source_sha256')!=shared
            or em.get('unit')!='V' or em.get('rate_hz')!=50 or em.get('filter_hz')!=[.5,8.] or len(em.get('channels',[]))!=40):raise ValueError('qualified40-channel numericalEEG required')
    ep=member(eeg_directory,'eeg.npz',em['files']['eeg.npz'])
    with np.load(ep,allow_pickle=False) as archive:
        eeg=archive['predicted_eeg_volts'];times=archive['query_seconds'];supported=archive['supported']
    if (times.ndim!=1 or eeg.shape!=(len(times),40) or supported.shape!=times.shape or supported.dtype!=bool
            or not len(times) or len(times)!=em['total_samples']
            or not np.isfinite(times).all() or not np.allclose(np.diff(times),.02,atol=1e-7)
            or abs(times[0]-em['source_offset_seconds'])>1e-8
            or times[-1]>em['source_offset_seconds']+em['duration_seconds']+1e-6
            or not np.isfinite(eeg[supported]).all() or not np.isnan(eeg[~supported]).all()
            or int(supported.sum())!=em['supported_samples']):raise ValueError('EEG clock, mask or values invalid')
    scores=Timeline.model_validate_json((scores_directory/'timeline.json').read_text())
    if (scores.source_sha256!=shared or scores.input_format!=presentation['derivative']['media_kind']
            or scores.duration_seconds>presentation['duration_seconds']+1e-3):raise ValueError('scores and neuralbranches use different presentations')
    if presentation['narration_generated'] and any(q.value is not None and (
            q.evidence_status!='experimental_transfer' or q.validation_status!='exploratory')
            for point in scores.points for q in point.qualities.values()):
        raise ValueError('generated narration requires experimental-transfer quality evidence')
    sm=json.loads((scores_directory/'manifest.json').read_text())
    member(scores_directory,sm['quality_timeline']['file'],sm['quality_timeline']['sha256'])
    expected_native=file_sha256(native_directory/'manifest.json');expected_eeg=file_sha256(eeg_directory/'manifest.json')
    # New fusion provenance must bind both numerical branches, never a silent cache.
    if sm.get('native_manifest_sha256')!=expected_native or sm.get('eeg_manifest_sha256')!=expected_eeg:raise ValueError('fusion scores do not bind these neural predictions')
    context_info=sm['neural_context'];context_path=member(scores_directory,context_info['file'],context_info['sha256'])
    context=json.loads(context_path.read_text())
    if (context.get('schema')!='perception-numerical-neural-context-v1'
            or [r['second'] for r in context['points']]!=[p.second for p in scores.points]):raise ValueError('neural context ledger clock mismatch')
    used={name:sum(bool(r['used_by_quality_prediction'][name]) for r in context['points']) for name in ('predicted_eeg','predicted_cortex')}
    if used!=context_info['used_points']:raise ValueError('neural context usage counts changed')
    for point,row in zip(scores.points,context['points']):
        valid=any(q.value is not None for q in point.qualities.values())
        if row['quality_prediction_supported']!=valid or any(
                row['used_by_quality_prediction'][name] and not (valid and row['available'][name]) for name in used):
            raise ValueError('neural context usage inconsistent with quality support')
    optional_files={key:member(scores_directory,sm[key]['file'],sm[key]['sha256'])
        for key in ('benchmark_reference','concepts') if key in sm}
    output.mkdir(parents=True)
    for source,name in [(native_path,'fmri.npz'),(native_directory/'manifest.json','native-tribe-manifest.json'),
            (ep,'eeg.npz'),(eeg_directory/'manifest.json','eeg-manifest.json'),
            (scores_directory/'timeline.json','scores.json'),(scores_directory/'manifest.json','scores-manifest.json'),
            (context_path,'neural-context.json'),
            (narration_directory/'manifest.json','presentation-manifest.json'),
            (surface_directory/'surface.npz','surface.npz'),(surface_directory/'manifest.json','surface-manifest.json')]:
        shutil.copyfile(source,output/name)
        if file_sha256(source)!=file_sha256(output/name):raise ValueError('source copy changed')
    for path in optional_files.values():
        shutil.copyfile(path,output/path.name)
        if file_sha256(path)!=file_sha256(output/path.name):raise ValueError('score evidence copy changed')
    descriptors={'predicted_eeg_volts':buffer(eeg,output,'eeg-volts.bin'),
        'eeg_query_seconds':buffer(times,output,'eeg-query-seconds.bin'),
        'eeg_supported':buffer(supported.astype(np.uint8),output,'eeg-supported.bin'),
        'predicted_cortical_response':buffer(cortex,output,'cortical-response.bin'),
        'native_segment_starts':buffer(starts,output,'cortical-starts.bin'),
        'native_segment_durations':buffer(durations,output,'cortical-durations.bin')}
    mesh={name:buffer(array,output,'surface-'+name+'.bin') for name,array in arrays.items()}
    response={'schema':'perception-numerical-brain-bundle-v1','complete':True,
        'original_source':presentation['parent'],'presented_source':presentation['derivative'],
        'input_format':presentation['parent']['media_kind'],'presentation_mode':presentation['mode'],
        'original_duration_seconds':presentation['duration_seconds'] if presentation['original_clock_preserved'] else None,
        'presentation_duration_seconds':presentation['duration_seconds'],
        'generated_narration':presentation['narration_generated'],'same_presented_content_for_neural_branches':True,
        'eeg':{'unit':'V','channels':em['channels'],'rate_hz':50,'filter_hz':[.5,8.],'target':em['target'],
            'clock_basis':'decoded presentation audio sample time plus documented source offset',
            'source_offset_seconds':em['source_offset_seconds'],'audio_duration_seconds':em['duration_seconds'],
            'values':descriptors['predicted_eeg_volts'],'query_seconds':descriptors['eeg_query_seconds'],
            'supported':descriptors['eeg_supported'],'unsupported_encoding':'NaN in binary; mask0; JSONclients must use null',
            'scientific_accuracy_on_this_lesson_established':False},
        'fmri':{'values':descriptors['predicted_cortical_response'],'starts':descriptors['native_segment_starts'],
            'durations':descriptors['native_segment_durations'],'native_segments':native['segments'],
            'neuro_offset_seconds':native.get('neuro_offset_seconds'),'values_unit':'native checkpoint output units',
            'clock_basis':'unchanged native TRIBE segment starts/durations; any native response tail beyond content extent is retained',
            'native_start_seconds':float(starts[0]),'native_stop_seconds':float((starts+durations).max()),
            'post_presentation_tail_seconds':max(0.,float((starts+durations).max())-presentation['duration_seconds']),
            'observed_scanner_data':False,'display_interpolation_applied':False,
            'native_bytes_preserved':True,'weights_changed':False},
        'surface':{'mesh':'fsaverage5','buffers':mesh,'hemisphere_columns':{'left':[0,10242],'right':[10242,20484]},
            'indexing_basis':'pinned upstream projector and synthetic index qualification','geometry_role':geometry['geometry_role'],
            'checkpoint_anatomy_independently_validated':False,'named_regions_provided':False},
        'scores':{'file':'scores.json','sha256':file_sha256(output/'scores.json'),
            'clock_basis':scores.clock_basis,'duration_seconds':scores.duration_seconds,'context_seconds':44,'query_step_seconds':1,
            'unanalyzed_presentation_tail_seconds':max(0.,presentation['duration_seconds']-scores.duration_seconds),
            'neural_input_policy':'Frozen predicted neural blocks are consumed only where the per-point ledger marks them available and used; other slots are masked.',
            'neural_context':{'file':'neural-context.json','sha256':context_info['sha256'],'used_points':used},
            'evidence_files':{key:{'file':path.name,'sha256':file_sha256(output/path.name)} for key,path in optional_files.items()},
            'supervision':'human attention labels and AI content-review proxies; experimental transfer'},
        'causal_realtime_accuracy_established':False,'participant_recordings_included':False,
        'scientific_accuracy_validated':False,'measured_fMRI_superiority_tested':False,
        'educational_outcome_improvement_demonstrated':False,
        'model_weights_included':False,'remote_calls_by_exporter':0,
        'provenance':{'eeg_manifest_sha256':expected_eeg,'native_manifest_sha256':expected_native,
            'scores_manifest_sha256':file_sha256(scores_directory/'manifest.json'),
            'presentation_manifest_sha256':file_sha256(narration_directory/'manifest.json')},
        'files':{p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(response,indent=2,allow_nan=False)+'\n');return response
