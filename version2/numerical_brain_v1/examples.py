"""Cached-only five-format execution. No example may fall back to cloud calls."""
import argparse
import json
from pathlib import Path

from brain.worker_contract import PublicMediaRequest,file_sha256,validate_media
from .pipeline import run,make_composite,preflight
from .presentation_batch import bind_local_content

ROOT=Path(__file__).resolve().parents[1]


def plan():
    cases=json.loads((ROOT.parent/'.local/brain-fixtures-multimodal-v1/manifest.json').read_text())['cases']
    rows={}
    for name in ('video','audio'):
        row=cases[name];request=PublicMediaRequest.model_validate_json(Path(row['request']).read_text())
        source=Path(row['source']);validate_media(source,request)
        content=ROOT.parent/f'.local/brain-runs/full-{name}-qualification-v1'
        m=json.loads((content/'manifest.json').read_text())
        if m['source']!=request.model_dump():raise ValueError('original source and cached content disagree')
        audio=json.loads((content/'audio/manifest.json').read_text())
        speech=ROOT/'.local/numerical-brain-v1/speech-examples-r1/native'
        sm=json.loads((speech/'manifest.json').read_text())
        if sm['audio_source_sha256']!=audio['derivative']['sha256']:raise ValueError('native speech feature cache mismatch')
        rows[name]={'source':str(source),'request':row['request'],'cached_content_root':str(content),
            'native_manifest_sha256':file_sha256(content/'tribe/manifest.json'),'speech_cache':str(speech),
            'replay_name':'example_'+name}
    narrated=json.loads((ROOT/'.local/numerical-brain-v1/narrated-inputs-r1/manifest.json').read_text())
    for row in narrated['records']:
        name=row['name'];rows[name]={'source':cases[name]['source'],'request':cases[name]['request'],
            'cached_presentation_root':row['presentation_directory'],'local_input':row,
            'speech_cache':str(ROOT/'.local/numerical-brain-v1/speech-examples-r1'/name)}
    return rows


def execute(formats,*,replay_root=None,narrated_native_root=None,output=None,composite=None,fusion_model=None):
    jobs=plan();output=Path(output or ROOT/'.local/numerical-brain-v1/numerical-examples-r1')
    composite=Path(composite or ROOT/'.local/numerical-brain-v1/composite-model-r1')
    if not formats or len(set(formats))!=len(formats) or set(formats)-set(jobs):raise ValueError('unique known formats required')
    if not composite.exists():
        make_composite(Path(fusion_model or ROOT/'.local/numerical-brain-v1/fusion-deployment-r2'),ROOT/'runs/endpoint-v1',
            ROOT/'.local/numerical-brain-v1/waveform-parity-r3/report.json',
            ROOT.parent/'.local/brain-runs/native-surface-qualification-v2',composite)
    preflight(composite);identity=file_sha256(composite/'manifest.json')
    output.mkdir(parents=True,exist_ok=True)
    for name in formats:
        row=jobs[name];request=PublicMediaRequest.model_validate_json(Path(row['request']).read_text())
        destination=output/name
        if destination.exists():
            if not (destination/'manifest.json').is_file():raise ValueError('incomplete prior example preserved; choose deliberate recovery')
            saved=json.loads((destination/'manifest.json').read_text())
            if (not saved.get('complete') or saved['source']!=request.model_dump()
                    or saved['composite_manifest_sha256']!=identity
                    or saved['bundle_sha256']!=file_sha256(destination/'bundle/manifest.json')):raise ValueError('saved example identity changed')
            continue
        if name in {'video','audio'}:
            if replay_root is None:raise ValueError('qualified saved native replay required; no cloud fallback')
            native_audio=Path(replay_root)/row['replay_name'];content=Path(row['cached_content_root']);presentation=None
        else:
            if narrated_native_root is None:raise ValueError('new narrated native capture required; no silent cache fallback')
            batch=Path(narrated_native_root);native=batch/'native'/name
            native_audio=native if name=='text' else batch/'audio-replay'/name
            presentation=Path(row['cached_presentation_root']);content=output/(name+'-content-cache')
            if not content.exists():bind_local_content(row['local_input'],native,content)
        audio_meta=json.loads((native_audio/'manifest.json').read_text())
        if not audio_meta.get('hidden_features') or not (audio_meta.get('hidden_capture_native_outputs_bitwise_equal') or audio_meta.get('replay_qualification',{}).get('passed')):
            raise ValueError('qualified native audio features absent; no paid fallback permitted')
        run(Path(row['source']),request,destination,'numerical-example-'+name,composite_model=composite,
            cached_content_root=content,cached_audio_native_root=native_audio,cached_presentation_root=presentation,
            speech_cache=Path(row['speech_cache']))
        print(json.dumps({'example_complete':name}),flush=True)
    results=[]
    for name in jobs:
        destination=output/name
        if not (destination/'manifest.json').is_file():continue
        m=json.loads((destination/'manifest.json').read_text());b=json.loads((destination/'bundle/manifest.json').read_text())
        if (not m.get('complete') or not b.get('complete') or m['composite_manifest_sha256']!=identity
                or m['bundle_sha256']!=file_sha256(destination/'bundle/manifest.json')):
            raise ValueError('existing example is incomplete or bound to a different artifact')
        results.append({'input_format':name,'complete':m['complete'],'pipeline_manifest_sha256':file_sha256(destination/'manifest.json'),
            'bundle_manifest_sha256':file_sha256(destination/'bundle/manifest.json'),'native_cortex_preserved':b['fmri']['native_bytes_preserved'],
            'eeg_channels':len(b['eeg']['channels']),'eeg_rate_hz':b['eeg']['rate_hz'],
            'neural_usage':b['scores']['neural_context']['used_points'],'presentation_mode':b['presentation_mode']})
    report={'schema':'perception-numerical-five-format-examples-v1','complete':len(results)==5,'examples':results,
        'composite_manifest_sha256':identity,'paid_calls':0,'participant_data_sent_to_cloud':False,
        'engineering_qualification_not_neural_accuracy':True}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--formats',nargs='+',default=['video','audio','text','image','pdf'])
    parser.add_argument('--replay-root',type=Path);parser.add_argument('--narrated-native-root',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--composite',type=Path);parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--fusion-model',type=Path)
    a=parser.parse_args()
    print(json.dumps(plan() if a.plan_only else execute(a.formats,replay_root=a.replay_root,narrated_native_root=a.narrated_native_root,
        output=a.output,composite=a.composite,fusion_model=a.fusion_model),indent=2))
