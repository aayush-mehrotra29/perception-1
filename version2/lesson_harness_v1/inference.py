"""Local lesson inference. No upload of neural summaries or local model weights."""
import argparse
import importlib.util
import json
import os
import subprocess
import shutil
from pathlib import Path

import numpy as np

from brain.edugage_targets import guard
from brain.fusion_data import prepare_native,load_grid
from brain.worker_contract import file_sha256
from . import QUALITIES, LEVELS, MODALITIES, VERSION
from .artifact import preflight
from .auxiliary import assemble
from .bundle import write_bundle
from .contract import QualityValue,TimelinePoint,Timeline
from .features import transform,load_transforms
from .model import load_network,predict
from .source import binding,references
from .windows import pooled_windows,flags


def preflight_runtime():
    from brain.fusion_inference import torch_python
    executable=torch_python()
    probe=subprocess.run([executable,'-B','-c','import torch, numpy, pydantic; from lesson_harness_v1.model import build_network; build_network(170)'],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=60,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
    if probe.returncode:raise RuntimeError('qualified local lesson runtime unavailable; configure BRAIN_TORCH_PYTHON')
    return executable


def forecast(blocks,duration,networks,states,heads,source,artifact_sha256,reference_population,
             auxiliary=None,concept_occurrences=()):
    """Pure forecast core; callers qualify data/artifacts and enforce resources first."""
    if len(networks)!=3 or set(heads)!=set(QUALITIES):
        raise ValueError('three fixed seed models and all five head statuses required')
    points=[]
    for seconds,means,masks in pooled_windows(blocks,duration):
        if auxiliary is not None:
            extra=auxiliary(seconds)
            if set(extra)&set(MODALITIES):raise ValueError('auxiliary data cannot replace native representations')
            means.update(extra)
        x=transform(means,states)
        usable=np.array([any(masks[i,MODALITIES.index(n)] for n in {'text','audio','video'}&set(states)) for i in range(len(seconds))])
        indices=np.flatnonzero(usable)
        probabilities=None
        if len(indices):
            probabilities=np.stack([predict(network,x[indices]) for network in networks]).astype(float)
            if not np.isfinite(probabilities).all():raise ValueError('nonfinite ensemble predictions')
        lookup={int(row):i for i,row in enumerate(indices)}
        for row,second in enumerate(seconds):
            active=[n for i,n in enumerate(MODALITIES) if masks[row,i] and n in states]
            refs=references(source,int(second)) if usable[row] else []
            qualities={}
            for h,name in enumerate(QUALITIES):
                info=heads[name]
                reason=('incomplete_context' if second<44 else 'no_supported_content_modality') if not usable[row] else None
                if info['status']!='fitted':reason=info.get('reason','head_not_fitted')
                if not refs and reason is None:reason='missing_source_reference'
                if reason is not None:
                    qualities[name]=QualityValue(evidence_status='unavailable',reason=reason)
                    continue
                ps=probabilities[:,lookup[row],h]
                p=ps.mean(0);p=p/p.sum()
                score=float(p@LEVELS);origins=info['supervision_origins']
                status='human_supported' if 'human' in origins else 'synthetic_proxy'
                if source['input_format']!='video':status='experimental_transfer'
                qualities[name]=QualityValue(distribution=p.tolist(),value=score,
                    ensemble_sd=float(np.std(ps@LEVELS)),evidence_status=status,supervision_origins=origins,
                    validation_status=info['validation_status'] if source['input_format']=='video' else 'exploratory',
                    training_label_resolution_seconds=info.get('training_label_resolution_seconds'))
            has_prediction=any(q.value is not None for q in qualities.values())
            points.append(TimelinePoint(second=int(second),context_start=int(second)-44 if has_prediction else None,
                context_stop=int(second) if has_prediction else None,qualities=qualities,
                active_modalities=active, references=refs,
                concept_ids=sorted({c['id'] for c in concept_occurrences if c['start']<second and c['stop']>max(0,second-44)})))
    return Timeline(source_sha256=source['source_sha256'],artifact_sha256=artifact_sha256,
        input_format=source['input_format'],duration_seconds=duration,clock_basis=source['clock_basis'],
        reference_population=reference_population,points=points,flags=flags(points))


def checked_cached_grid(work,native,source_sha256,duration):
    meta,blocks=load_grid(work)
    if (meta['source_sha256']!=source_sha256 or meta['duration_seconds']!=duration
            or meta.get('frequency_hz')!=2
            or meta['native_manifest_sha256']!=file_sha256(Path(native)/'manifest.json')):
        raise ValueError('cached grid media, duration or native model identity mismatch')
    return meta,blocks


def predict_native(native,model,output,*,duration,input_format,source_sha256,lineage_directory=None,
                   feature_sources=None,concept_occurrences=(),cached_grid=None):
    native,model,output=map(Path,(native,model,output))
    guard(output)
    if output.exists():raise ValueError('preserve existing output; choose a new destination')
    meta=preflight(model)
    if importlib.util.find_spec('torch') is None:
        executable=preflight_runtime()
        args=[executable,'-B','-m','lesson_harness_v1.inference','--native',str(native.resolve()),
              '--model',str(model.resolve()),'--output',str(output.resolve()),'--duration',str(duration),
              '--input-format',input_format,'--source-sha256',source_sha256,
              '--concepts-json',json.dumps(list(concept_occurrences))]
        if lineage_directory is not None:args+=['--lineage-directory',str(Path(lineage_directory).resolve())]
        if cached_grid is not None:args+=['--cached-grid',str(Path(cached_grid).resolve())]
        if feature_sources is not None:args+=['--feature-sources-json',json.dumps(feature_sources)]
        result=subprocess.run(args,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=1800,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode:raise RuntimeError('local lesson inference failed; partial outputs preserved: '+result.stderr[-1000:])
        return json.loads((output/'manifest.json').read_text())
    native_meta=json.loads((native/'manifest.json').read_text())
    source=binding(native_meta,source_sha256,input_format,duration,lineage_directory)
    work=Path(cached_grid) if cached_grid is not None else output.parent/(output.name+'-lesson-work')
    if cached_grid is None:prepare_native(native,duration,work)
    grid_meta,blocks=checked_cached_grid(work,native,source['native_source_sha256'],duration)
    import torch
    torch.set_num_threads(1)
    states=load_transforms(model/'transforms')
    networks=[load_network(model/Path(member['manifest']).parent) for member in meta['ensemble']]
    def auxiliary(seconds):
        guard(output)
        if feature_sources is None:return {'progress':seconds[:,None]/duration}
        return assemble(feature_sources,seconds,source_sha256,duration)
    timeline=forecast(blocks,duration,networks,states,meta['heads'],source,
                      file_sha256(model/'manifest.json'),meta['reference_population'],auxiliary,concept_occurrences)
    guard(output)
    result=write_bundle(output,timeline,native,evaluation_status='experimental',lineage_directory=lineage_directory)
    concepts={'schema':'perception-lesson-concepts-v1','occurrences':list(concept_occurrences),
              'timing':'source semantic spans; not precise recognition times'}
    (output/'concepts.json').write_text(json.dumps(concepts,indent=2,allow_nan=False)+'\n')
    result['concepts']={'file':'concepts.json','sha256':file_sha256(output/'concepts.json')}
    result['feature_grid_manifest_sha256']=file_sha256(work/'manifest.json')
    result['existing_feature_grid_reused']=cached_grid is not None
    shutil.copyfile(model/'benchmark.json',output/'benchmark.json')
    if file_sha256(output/'benchmark.json')!=meta['files']['benchmark.json']:
        raise ValueError('benchmark export checksum changed')
    result['benchmark_reference']={'file':'benchmark.json','sha256':meta['files']['benchmark.json']}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['native','model','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--duration',type=float,required=True)
    parser.add_argument('--input-format',choices=['video','audio','text','pdf','image'],required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--lineage-directory',type=Path)
    parser.add_argument('--cached-grid',type=Path)
    parser.add_argument('--feature-sources-json')
    parser.add_argument('--concepts-json',default='[]')
    a=parser.parse_args()
    print(json.dumps(predict_native(a.native,a.model,a.output,duration=a.duration,input_format=a.input_format,
        source_sha256=a.source_sha256,lineage_directory=a.lineage_directory,
        feature_sources=json.loads(a.feature_sources_json) if a.feature_sources_json else None,
        concept_occurrences=json.loads(a.concepts_json),cached_grid=a.cached_grid)))

