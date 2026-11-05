"""Bounded local transport helpers; no remote invocation or credit mutation."""
import io
import json
from pathlib import Path
import zipfile

from brain.edugage_targets import guard
from brain.modal_client import unpack_result
from brain.worker_contract import PublicMediaRequest,file_sha256,validate_media
from .cortical_replay import prepare_request,materialize as materialize_replay

NAMES=('text','image','pdf')


def prepare(manifest_path):
    meta=json.loads(Path(manifest_path).read_text())
    if meta.get('schema')!='perception-narrated-input-batch-v1' or not meta.get('complete'):
        raise ValueError('complete narrated local input bundle required')
    rows=meta['records']
    if [r['name'] for r in rows]!=list(NAMES):raise ValueError('fixed ordered narration inputs required')
    items=[]
    for row in rows:
        request=PublicMediaRequest.model_validate(row['request']);media=Path(row['media']);validate_media(media,request)
        if row['media_sha256']!=request.sha256:raise ValueError('input manifest media changed')
        if request.media_kind!=('audio' if row['name']=='text' else 'video'):raise ValueError('narrated media type mismatch')
        items.append({'name':row['name'],'request':request.model_dump(),'media':media.read_bytes()})
    if sum(len(i['media']) for i in items)>64*1024**2:raise ValueError('transport bound exceeded')
    return items


def subset(archive,prefix,exclude=()):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as out:
        for name in archive.namelist():
            if name.startswith(prefix):
                local=name[len(prefix):]
                if local and local not in exclude:out.writestr(local,archive.read(name))
    return buffer.getvalue()


def materialize(payload,items,output):
    output=Path(output);guard(output)
    if output.exists():raise ValueError('preserve prior native batch')
    if len(payload)>256*1024**2:raise ValueError('batch transport bound exceeded')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=archive.namelist()
        if (len(names)!=len(set(names)) or 'manifest.json' not in names
                or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n or n.endswith('/') for n in names)
                or any(n!='manifest.json' and not n.startswith(('native/text/','native/image/','native/pdf/',
                    'audio-replay/qualification/','audio-replay/image/','audio-replay/pdf/','audio-replay/receipt.json')) for n in names)
                or sum(i.file_size for i in archive.infolist())>256*1024**2):raise ValueError('unsafe or excessive batch archive')
        report=json.loads(archive.read('manifest.json'))
        if (report.get('schema')!='perception-narrated-native-batch-v1' or not report.get('complete')
                or report['sources']!={i['name']:i['request']['sha256'] for i in items} or not report.get('replay_qualified')):
            raise ValueError('native batch source qualification mismatch')
        output.mkdir(parents=True)
        for item in items:
            root=output/'native'/item['name']
            unpack_result(subset(archive,'native/'+item['name']+'/'),root,'tribe',
                PublicMediaRequest.model_validate(item['request']),require_hidden=True,publication_guard=guard)
            if file_sha256(root/'manifest.json')!=report['native_manifests'][item['name']]:raise ValueError('native manifest identity changed')
        replay_request=prepare_request({'image':output/'native/image','pdf':output/'native/pdf'},output/'native/text')
        dense=[r['name']+'/'+row['file'] for r in replay_request['sources'] for row in r['dense']]
        materialize_replay(subset(archive,'audio-replay/',exclude=dense),replay_request,output/'audio-replay')
        # The verified native/replay manifests are authoritative; unexpected
        # archive files cannot silently enter the published artifact.
        allowed={'manifest.json','audio-replay/receipt.json'}
        for folder in [output/'native'/n for n in NAMES]+[output/'audio-replay'/n for n in ('qualification','image','pdf')]:
            allowed.update(str(p.relative_to(output)) for p in folder.iterdir() if p.is_file())
        if set(names)!=allowed:raise ValueError('unexpected batch members')
        (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def bind_local_content(row,native,output):
    """Join verified local decoding and fresh native outputs without copying them."""
    native,output=Path(native).resolve(),Path(output);guard(output)
    if output.exists():raise ValueError('preserve prior content binding')
    request=PublicMediaRequest.model_validate(row['request']);validate_media(Path(row['media']),request)
    nm=json.loads((native/'manifest.json').read_text())
    if nm['source']!=request.model_dump():raise ValueError('native and local presentation identities differ')
    audio=Path(row['audio_directory']).resolve();am=json.loads((audio/'manifest.json').read_text())
    if am['parent']!=request.model_dump() or not am['audio_clock_continuity_checked']:
        raise ValueError('decoded audio belongs to another presentation')
    validate_media(audio/'audio.wav',PublicMediaRequest.model_validate(am['derivative']))
    spec=json.loads(Path(row['feature_sources']).read_text())
    if spec['source_sha256']!=request.sha256 or spec['media_duration']!=row['analysis_duration_seconds']:
        raise ValueError('local features source or clock mismatch')
    output.mkdir(parents=True);(output/'audio').symlink_to(audio,target_is_directory=True)
    (output/'tribe').symlink_to(native,target_is_directory=True)
    spec['tribe']=str(native)
    (output/'feature-sources.json').write_text(json.dumps(spec,indent=2)+'\n')
    result={'schema':'perception-numerical-local-content-v1','complete':True,'source':request.model_dump(),
        'native_inference_source':nm['source'],'native_manifest_sha256':file_sha256(native/'manifest.json'),
        'duration_seconds':spec['media_duration'],'supplemental_policy':'cached_only',
        'supplemental_features':{k:'missing_masked' for k in ('sam_audio','sam_visual','semantics')},
        'new_SAM_or_semantic_API_calls':0,'bound_from_checked_narrated_batch':True}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result
