"""Replay authenticated five-format content caches into new lesson bundles."""
import argparse
import json
from pathlib import Path
import time

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .concepts import collect
from .inference import predict_native


def build(model, source_root, output):
    model, source_root, output = map(Path, (model, source_root, output))
    guard(output)
    if output.exists(): raise ValueError('preserve prior examples')
    output.mkdir(parents=True)
    results=[]
    for kind in ['video','audio','text','pdf','image']:
        guard(output); started=time.monotonic()
        root=source_root/f'full-{kind}-qualification-v1'
        pipeline=json.loads((root/'manifest.json').read_text())
        native=json.loads((root/'tribe/manifest.json').read_text())
        sha=pipeline['source']['sha256']
        grid=source_root/'fusion-v2-qualification'/kind/'prediction-fusion-work'
        grid_meta=json.loads((grid/'manifest.json').read_text());duration=grid_meta['duration_seconds']
        lineage=root/'audio' if kind=='audio' else root/'presentation' if kind in {'pdf','image'} else None
        if kind=='video':
            spec=json.loads((root/'feature-sources.json').read_text())
        else:
            spec={'source_sha256':sha,'media_duration':duration}
            if (root/'semantics/manifest.json').exists(): spec['semantics']=str((root/'semantics').resolve())
            if (root/'sam_audio/manifest.json').exists():
                spec.update(sam_audio=str((root/'sam_audio').resolve()),audio_derivative=str((root/'audio').resolve()))
            if (root/'sam_visual/manifest.json').exists():
                spec.update(sam_visual=str((root/'sam_visual').resolve()),frames=str((root/'frames').resolve()),
                            rendered_visual_source_sha256=native['source']['sha256'])
        concepts=collect(Path(spec['semantics']),sha) if 'semantics' in spec else []
        bundle=predict_native(root/'tribe',model,output/kind,duration=duration,input_format=kind,
            source_sha256=sha,lineage_directory=lineage,feature_sources=spec,concept_occurrences=concepts,cached_grid=grid)
        timeline=json.loads((output/kind/'timeline.json').read_text())
        if [p['second'] for p in timeline['points']]!=list(range(int(duration)+1)):
            raise ValueError('missing integer-second display positions')
        if any(q['value'] is not None for p in timeline['points'] if p['second']<44 for q in p['qualities'].values()):
            raise ValueError('warmup is not null')
        if (file_sha256(output/kind/'fmri.npz')!=native['array_sha256']
                or file_sha256(root/'tribe/fmri.npz')!=native['array_sha256']):
            raise ValueError('native cortical predictions changed')
        results.append({'input_format':kind,'source_sha256':sha,'bundle_sha256':file_sha256(output/kind/'manifest.json'),
            'duration_seconds':duration,'supported_points_by_quality':bundle['quality_timeline']['supported_points_by_quality'],
            'native_TRIBE_unchanged':True,'elapsed_seconds':time.monotonic()-started,
            'engineering_qualification_not_human_accuracy':True,'original_pipeline_manifest_sha256':file_sha256(root/'manifest.json')})
        print(kind+': bundle verified',flush=True)
    report={'schema':'perception-lesson-five-format-examples-v1','complete':True,'examples':results,
        'artifact_sha256':file_sha256(model/'manifest.json'),'paid_calls':0,'participant_data_sent_to_cloud':False,
        'source_media':'existing original project-authored educational fixtures; domain accuracy unvalidated'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['model','source_root','output']:p.add_argument(name,type=Path)
    a=p.parse_args();build(a.model,a.source_root,a.output)
