"""Composite numerical backend; paid content inference preserves existing ceilings."""
import json
from pathlib import Path

from brain.edugage_targets import guard
from brain.worker_contract import PublicMediaRequest,file_sha256,validate_media
from .narration import prepare

SHARED_STATE=Path(__file__).resolve().parents[2]/'.local'


def local_content(source,request,output,request_prefix,*,cached_native_root=None,supplemental_root=None):
    """Extract local descriptors; only foundation TRIBE may make a paid call.

    SAM and semantic features are optional, identity-checked existing caches.
    This deliberately avoids re-running paid supplementary models for demos.
    """
    from brain.audio_derivative import extract as extract_audio
    from brain.sensory import extract as extract_sensory
    from brain.modal_client import submit
    from brain.credentials import load_credentials
    source,output=Path(source),Path(output);validate_media(source,request);guard(output)
    if request.media_kind not in {'video','audio'}:raise ValueError('actual narrated or native media required')
    output.mkdir(parents=True,exist_ok=False)
    audio=extract_audio(source,request,output/'audio')
    audio_request=PublicMediaRequest.model_validate(audio['derivative'])
    native_media,native_request=(source,request) if request.media_kind=='video' else (output/'audio/audio.wav',audio_request)
    duration=audio['samples']/audio['sample_rate']
    spec={'schema':'perception-lesson-feature-sources-v1','source_sha256':request.sha256,
        'lecture':request.source_id,'media_duration':duration,'audio_derivative':str((output/'audio').resolve())}
    if request.media_kind=='video':
        sensory=extract_sensory(source,request,output/'sensory')
        duration=sensory['duration_seconds'];spec.update(media_duration=duration,sensory_directory=str((output/'sensory').resolve()))
    supplemental={name:'missing_masked' for name in ('sam_audio','sam_visual','semantics')}
    if supplemental_root is not None:
        cache=Path(supplemental_root);cm=json.loads((cache/'manifest.json').read_text())
        if cm['source']!=request.model_dump():raise ValueError('supplemental features belong to another presentation')
        for name in supplemental:
            if (cache/name/'manifest.json').is_file():
                spec[name]=str((cache/name).resolve());supplemental[name]='cached'
        if 'sam_audio' in spec:spec['audio_derivative']=str((cache/'audio').resolve())
        if 'sam_visual' in spec:spec['frames']=str((cache/'frames').resolve())
    if cached_native_root is not None:
        cached_native_root=Path(cached_native_root).resolve()
        nm=json.loads((cached_native_root/'manifest.json').read_text())
        if nm['source']!=native_request.model_dump():raise ValueError('native cortex cache belongs to another presentation')
        (output/'tribe').symlink_to(cached_native_root,target_is_directory=True)
    else:
        load_credentials(SHARED_STATE/'brain-credentials.json')
        kwargs={'app_name':'perception-brain-v2','require_hidden':True} if request.media_kind=='audio' else {}
        submit('tribe',native_media,native_request,output/'tribe',request_prefix+'-tribe',state=SHARED_STATE,
            reservation_microdollars=2000000 if request.media_kind=='audio' else 4000000,**kwargs)
    spec['tribe']=str((output/'tribe').resolve())
    (output/'feature-sources.json').write_text(json.dumps(spec,indent=2)+'\n')
    result={'schema':'perception-numerical-local-content-v1','complete':True,'source':request.model_dump(),
        'native_inference_source':native_request.model_dump(),'native_manifest_sha256':file_sha256(output/'tribe/manifest.json'),
        'duration_seconds':duration,'supplemental_features':supplemental,'supplemental_policy':'cached_only',
        'new_SAM_or_semantic_API_calls':0,'shared_credit_state':str(SHARED_STATE),
        'duration_policy':'complete local sensory coverage' if request.media_kind=='video' else 'decoded audio sample count'}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def preflight(composite_model):
    root=Path(composite_model);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema')!='perception-numerical-composite-model-v1' or not manifest.get('complete'):
        raise ValueError('fitted numerical composite model required')
    required={'fusion_model','waveform_endpoint','waveform_parity','surface'}
    if set(manifest['components'])!=required:raise ValueError('four pinned model components required')
    for name,record in manifest['components'].items():
        path=Path(record['path']);pin=path/record['identity_file'] if path.is_dir() else path
        if not pin.is_file() or file_sha256(pin)!=record['sha256']:raise ValueError('composite component identity changed: '+name)
    from .fusion_inference import preflight as fusion_preflight
    from .waveform import preflight as wave_preflight
    from brain.surface import load_surface
    fusion_preflight(Path(manifest['components']['fusion_model']['path']))
    wave_preflight(Path(manifest['components']['waveform_endpoint']['path']))
    if not json.loads(Path(manifest['components']['waveform_parity']['path']).read_text()).get('passed'):
        raise ValueError('waveform feature parity must pass before inference')
    load_surface(Path(manifest['components']['surface']['path']))
    return manifest


def make_composite(fusion_model,waveform_endpoint,waveform_parity,surface,output):
    output=Path(output);guard(output)
    if output.exists():raise ValueError('preserve previous composite descriptor')
    records={}
    for name,path,identity in [('fusion_model',fusion_model,'manifest.json'),('waveform_endpoint',waveform_endpoint,'forecast-lock.json'),
            ('waveform_parity',waveform_parity,None),('surface',surface,'manifest.json')]:
        path=Path(path).resolve();pin=path/identity if identity else path
        records[name]={'path':str(path),'identity_file':identity,'sha256':file_sha256(pin)}
    result={'schema':'perception-numerical-composite-model-v1','complete':True,'components':records,
        'description':'Frozen speech-EEG and native TRIBE predictions feed the experimental neural-feature quality fusion model.',
        'participant_recordings_sent_to_cloud':False,'scientific_accuracy_on_uploaded_lessons_established':False}
    output.mkdir(parents=True);(output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    preflight(output);return result


def run(source,request,output,request_prefix,*,composite_model,caption_records=None,voice='Samantha (English (US))',
        narration_rate=180,cached_content_root=None,cached_audio_native_root=None,cached_grid=None,speech_cache=None,
        cached_native_root=None,supplemental_root=None,cached_presentation_root=None):
    source,output=Path(source),Path(output);validate_media(source,request);guard(output)
    model=preflight(composite_model)
    from lesson_harness_v1.inference import preflight_runtime
    preflight_runtime()
    if output.exists():raise ValueError('preserve previous numerical pipeline')
    output.mkdir(parents=True);stage='shared_presentation'
    try:
        if cached_presentation_root is None:
            presentation=prepare(source,request,output/'presentation',caption_records=caption_records,voice=voice,rate=narration_rate)
        else:
            cached=Path(cached_presentation_root).resolve();presentation=json.loads((cached/'manifest.json').read_text())
            if (presentation.get('schema')!='numerical-narrated-presentation-v1' or not presentation.get('complete')
                    or presentation['parent']!=request.model_dump()):raise ValueError('cached narration belongs to another original source')
            validate_media(Path(presentation['media_path']),PublicMediaRequest.model_validate(presentation['derivative']))
            (output/'presentation').symlink_to(cached,target_is_directory=True)
        presented=PublicMediaRequest.model_validate(presentation['derivative']);media=Path(presentation['media_path'])
        stage='frozen_content_and_cortex'
        if cached_content_root is None:
            content=output/'content';content_result=local_content(media,presented,content,request_prefix+'-content',
                cached_native_root=cached_native_root,supplemental_root=supplemental_root)
        else:
            content=Path(cached_content_root);content_result=json.loads((content/'manifest.json').read_text())
            if content_result['source']!=presented.model_dump():raise ValueError('cached content belongs to another presentation')
            native_meta=json.loads((content/'tribe/manifest.json').read_text())
            if content_result.get('native_manifest_sha256') and content_result['native_manifest_sha256']!=file_sha256(content/'tribe/manifest.json'):
                raise ValueError('cached native manifest identity changed')
        if presented.media_kind=='video':
            spec=json.loads((content/'feature-sources.json').read_text());duration=spec['media_duration'];lineage=None
        elif presented.media_kind=='audio':
            audio_meta=json.loads((content/'audio/manifest.json').read_text());duration=audio_meta['samples']/audio_meta['sample_rate'];lineage=content/'audio'
            spec=json.loads((content/'feature-sources.json').read_text()) if (content/'feature-sources.json').is_file() else {'source_sha256':presented.sha256,'media_duration':duration}
            if (content/'semantics/manifest.json').exists():spec['semantics']=str((content/'semantics').resolve())
            if (content/'sam_audio/manifest.json').exists():spec.update(sam_audio=str((content/'sam_audio').resolve()),audio_derivative=str((content/'audio').resolve()))
        else:raise ValueError('neural branches require actual timed audio/video presentation')
        audio_meta=json.loads((content/'audio/manifest.json').read_text());audio_path=content/'audio/audio.wav'
        audio_request=PublicMediaRequest.model_validate(audio_meta['derivative'])
        if audio_meta['parent']!=presented.model_dump():raise ValueError('audio branch is not decoded from the shared presentation')
        validate_media(audio_path,audio_request)
        offset=audio_meta['first_native_audio_seconds'] if presented.media_kind=='video' else 0.
        stage='audio_native_features'
        native_meta=json.loads((content/'tribe/manifest.json').read_text())
        qualified_hidden=bool(native_meta.get('hidden_features')) and (native_meta.get('hidden_capture_native_outputs_bitwise_equal') or native_meta.get('replay_qualification',{}).get('passed'))
        if native_meta['source']['sha256']==audio_request.sha256 and qualified_hidden:audio_native=content/'tribe'
        elif cached_audio_native_root is not None:
            audio_native=Path(cached_audio_native_root)
            am=json.loads((audio_native/'manifest.json').read_text())
            if am['source']['sha256'] not in {audio_request.sha256,presented.sha256}:
                raise ValueError('audio hidden cache belongs to another decoded audio track')
            if not am.get('hidden_features') or not (am.get('hidden_capture_native_outputs_bitwise_equal') or am.get('replay_qualification',{}).get('passed')):
                raise ValueError('audio hidden cache is not numerically qualified')
        else:
            from brain.modal_client import submit
            audio_native=output/'audio-native'
            submit('tribe',audio_path,audio_request,audio_native,request_prefix+'-audio-native',state=SHARED_STATE,
                app_name='perception-brain-v2',require_hidden=True,reservation_microdollars=2000000)
        stage='numerical_eeg'
        from .waveform import predict as predict_waveform
        predict_waveform(audio_path,audio_native,output/'eeg',source_sha256=presented.sha256,source_offset_seconds=offset,
            endpoint_run=Path(model['components']['waveform_endpoint']['path']),parity_receipt=Path(model['components']['waveform_parity']['path']),
            audio_lineage=content/'audio/manifest.json',speech_cache=speech_cache)
        stage='neural_feature_fusion'
        from .fusion_inference import predict_native
        predict_native(content/'tribe',Path(model['components']['fusion_model']['path']),output/'scores',duration=duration,
            input_format=presented.media_kind,source_sha256=presented.sha256,eeg_directory=output/'eeg',
            lineage_directory=lineage,feature_sources=spec,cached_grid=cached_grid,
            experimental_transfer=bool(presentation['narration_generated']))
        stage='numerical_export'
        from .export import export
        bundle=export(output/'eeg',content/'tribe',output/'scores',Path(model['components']['surface']['path']),output/'bundle',
            narration_directory=output/'presentation',audio_lineage_directory=lineage)
        result={'schema':'perception-composite-content-pipeline-v1','complete':True,'source':request.model_dump(),
            'presented_source':presented.model_dump(),'composite_manifest_sha256':file_sha256(Path(composite_model)/'manifest.json'),
            'bundle':'bundle','bundle_sha256':file_sha256(output/'bundle/manifest.json'),
            'eeg':'eeg','scores':'scores','native_cortex_directory':str((content/'tribe').resolve()),
            'content_cache_reused':cached_content_root is not None,'caption_or_narration_is_simulated':presentation['narration_generated'],
            'same_presentation_for_neural_branches':True,'participant_data_sent_to_cloud':False,'paid_failures_automatically_retried':False}
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result
    except Exception as error:
        (output/'incomplete.json').write_text(json.dumps({'stage':stage,'error_type':type(error).__name__,
            'previous_outputs_preserved':True,'paid_failures_automatically_retried':False},indent=2)+'\n')
        raise
