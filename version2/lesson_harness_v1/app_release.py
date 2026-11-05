"""Verified browser-facing export; preserves predictions and labels their limits.

This adapter never fits a model, requests remote inference, or substitutes values
for unavailable heads. Native cortical arrays are exposed as a typed buffer for
JavaScript, alongside byte-identical NPZ and native timing metadata.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import LEVELS, QUALITIES
from .contract import Timeline


CATEGORIES = ('very_low', 'low', 'moderate', 'high', 'very_high')
DESCRIPTIONS = {
    'attention_difficulty': 'Predicted difficulty sustaining attention; higher means more difficulty.',
    'interest_engagement': 'Predicted interest in continuing; higher means more interest.',
    'mental_effort': 'Predicted processing demand; higher effort is not necessarily worse teaching.',
    'confusion_risk': 'Predicted risk of losing the explanatory thread.',
    'comprehension_difficulty': 'Predicted difficulty constructing understanding; not measured mastery.',
}


def checked_child(root, name, digest):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_sha256(path)!=digest:
        raise ValueError('bundle member identity mismatch')
    return path


def category(value):
    if value.distribution is None:
        return None
    # Fixed ordinal-mode rule. Ties resolve toward the lower numerical level.
    i=max(range(len(LEVELS)),key=lambda j:value.distribution[j])
    return {'id':CATEGORIES[i],'level':LEVELS[i],
            'probability':value.distribution[i],
            'meaning':'mode_of_predicted_ordinal_distribution_not_observed_student_state'}


def verify_bundle(root, require_all_heads=False):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if (manifest.get('schema')!='perception-lesson-bundle-v1'
            or manifest.get('participant_data_included') is not False
            or manifest.get('files_exclude_model_weights') is not True):
        raise ValueError('public lesson bundle required')
    paths={}
    for field in ('quality_timeline','concepts','benchmark_reference'):
        member=manifest[field]
        paths[member['file']]=checked_child(root,member['file'],member['sha256'])
    fmri=manifest['fmri']
    if fmri.get('TRIBE_weights_changed') is not False or fmri.get('observed_fmri') is not False:
        raise ValueError('unchanged predicted native TRIBE branch required')
    paths[fmri['file']]=checked_child(root,fmri['file'],fmri['sha256'])
    paths[fmri['native_manifest']]=checked_child(root,fmri['native_manifest'],fmri['native_manifest_sha256'])
    timeline=Timeline.model_validate_json(paths[manifest['quality_timeline']['file']].read_text())
    if timeline.source_sha256!=manifest['source_sha256'] or timeline.artifact_sha256!=manifest['artifact_sha256']:
        raise ValueError('bundle and timeline identity differ')
    counts={q:sum(p.qualities[q].value is not None for p in timeline.points) for q in QUALITIES}
    if counts!=manifest['quality_timeline']['supported_points_by_quality']:
        raise ValueError('reported supported counts differ')
    if require_all_heads and not all(counts.values()):
        raise ValueError('all five heads need supported predictions for this release')
    if any(v.value is not None for p in timeline.points if p.second<44 for v in p.qualities.values()):
        raise ValueError('incomplete-context predictions found')
    native=json.loads(paths[fmri['native_manifest']].read_text())
    if native.get('array_sha256')!=fmri['sha256']:
        raise ValueError('native array identity differs')
    segments=native['segments']
    starts=np.asarray([s['start'] for s in segments]);durations=np.asarray([s['duration'] for s in segments])
    if (not len(starts) or not np.isfinite(starts).all() or not np.isfinite(durations).all()
            or np.any(durations<=0) or np.any(starts[1:]<starts[:-1]+durations[:-1]-1e-8)
            or len({s['timeline'] for s in segments})!=1):
        raise ValueError('invalid native clock')
    with np.load(paths[fmri['file']],allow_pickle=False) as data:
        values=data['predicted_cortical_response']
        if (values.dtype.kind!='f' or values.dtype.itemsize not in (4,8)
                or values.ndim!=2 or not all(values.shape) or list(values.shape)!=native['shape']
                or len(values)!=len(segments) or not np.isfinite(values).all()):
            raise ValueError('unsupported cortical array')
    benchmark=json.loads(paths[manifest['benchmark_reference']['file']].read_text())
    if not benchmark.get('complete'):
        raise ValueError('complete benchmark report required')
    paths['manifest.json']=root/'manifest.json'
    if (root/'source-lineage.json').exists():
        digest=manifest['source_binding'].get('lineage_manifest_sha256')
        if not digest:
            raise ValueError('source lineage lacks identity')
        paths['source-lineage.json']=checked_child(root,'source-lineage.json',digest)
    return manifest,timeline,native,paths,counts


def export(source,destination,require_all_heads=False):
    source,destination=Path(source),Path(destination)
    guard(destination)
    if destination.exists():
        raise ValueError('preserve existing release; choose a new directory')
    manifest,timeline,native,paths,counts=verify_bundle(source,require_all_heads)
    before={name:file_sha256(path) for name,path in paths.items()}
    destination.mkdir(parents=True,exist_ok=False)
    for name,path in paths.items():
        shutil.copyfile(path,destination/name)
    with np.load(source/manifest['fmri']['file'],allow_pickle=False) as data:
        original=data['predicted_cortical_response']
        values=np.ascontiguousarray(original,dtype=original.dtype.newbyteorder('<'))
        values.tofile(destination/'cortical-response.bin')
        restored=np.fromfile(destination/'cortical-response.bin',dtype=values.dtype).reshape(values.shape)
        if restored.tobytes()!=values.tobytes():
            raise ValueError('browser cortical buffer differs')
        cortical={'file':'cortical-response.bin','sha256':file_sha256(destination/'cortical-response.bin'),
            'dtype':values.dtype.str,'typed_array':'Float32Array' if values.dtype.itemsize==4 else 'Float64Array',
            'byte_order':'little','layout':'C','shape':list(values.shape),'row_axis':'native_time',
            'column_axis':'native_cortical_vertex','segments':native['segments'],
            'spatial_metadata':native.get('spatial_metadata'),
            'named_brain_regions_provided':False,
            'spatial_vertex_order_qualified':native.get('spatial_metadata',{}).get('vertex_order_qualified',False),
            'observed_fmri':False,'TRIBE_weights_changed':False,'display_interpolation_applied':False}
    response={'schema':'perception-browser-brain-response-v1',
        'model_version':timeline.model_version,'artifact_sha256':timeline.artifact_sha256,
        'source_sha256':timeline.source_sha256,'input_format':timeline.input_format,
        'duration_seconds':timeline.duration_seconds,'clock_basis':timeline.clock_basis,
        'context_seconds':44,'query_step_seconds':1,
        'reference_population':timeline.reference_population,
        'psychological_resolution_validated_seconds':None,'observed_student_state':False,
        'definitions':DESCRIPTIONS,'category_levels':list(LEVELS),'category_names':list(CATEGORIES),
        'categorization_rule':'ordinal mode; ties choose lower level; not a second trained classifier',
        'timeline':[{'second':p.second,'context_start':p.context_start,'context_stop':p.context_stop,
            'active_modalities':p.active_modalities,'references':[r.model_dump() for r in p.references],
            'concept_ids':p.concept_ids,
            'qualities':{q:{**p.qualities[q].model_dump(),'category':category(p.qualities[q])} for q in QUALITIES}}
            for p in timeline.points],
        'flags':[f.model_dump() for f in timeline.flags],
        'fmri':cortical,'benchmark':manifest['benchmark_reference'],
        'quality_output_is_measured_attention_or_learning':False,
        'quality_output_is_neuroscience_ground_truth':False,
        'supervision_interpretation':{
            'human':'Supervised using available human labels; accuracy and temporal scope remain limited by the benchmark.',
            'synthetic':'AI content-review judgments for the assumed learner; not measured psychological states derived from EEG.',
            'experimental_transfer':'Input format outside the evaluated video domain; the original supervision origins still apply.'},
        'student_agents_executed':False}
    (destination/'app-response.json').write_text(json.dumps(response,indent=2,allow_nan=False)+'\n')
    for name,path in paths.items():
        if file_sha256(path)!=before[name] or file_sha256(destination/name)!=before[name]:
            raise ValueError('bundle changed during export')
    report={'schema':'perception-browser-release-check-v1','complete':True,
        'all_five_heads_supported':all(counts.values()),'supported_points_by_quality':counts,
        'native_npz_byte_identical':True,'native_manifest_byte_identical':True,
        'native_buffer_values_identical':True,'one_second_grid_verified':True,
        'warmup_null_verified':True,'quality_categories_are_display_only':True,
        'source_bundle_sha256':before['manifest.json'],'participant_data_included':False,
        'contains_model_weights':False,'remote_calls':0,
        'files':{p.name:file_sha256(p) for p in destination.iterdir() if p.is_file()}}
    (destination/'release-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    parser.add_argument('--require-all-heads',action='store_true');args=parser.parse_args()
    print(json.dumps(export(args.source,args.destination,args.require_all_heads),indent=2))
