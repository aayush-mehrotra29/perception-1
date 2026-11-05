"""Frontend/agent export with byte-preserved native TRIBE predictions."""
import json
import shutil
from pathlib import Path

import numpy as np

from brain.worker_contract import file_sha256
from .contract import Timeline


def write_bundle(destination, timeline, native_directory, *, evaluation_status, lineage_directory=None):
    if evaluation_status not in {"experimental", "human_evaluated", "synthetic_contract_test"}:
        raise ValueError("explicit evaluation status required")
    timeline = Timeline.model_validate(timeline)
    native_directory, destination = Path(native_directory), Path(destination)
    if destination.exists():raise ValueError("preserve existing bundle; choose a new destination")
    manifest_path=native_directory/'manifest.json';native_path=native_directory/'fmri.npz'
    manifest_hash=file_sha256(manifest_path)
    native=json.loads(manifest_path.read_text())
    if native.get('schema')!='perception-native-tribe-v1':
        raise ValueError('native TRIBE source required')
    from .source import binding
    bound = binding(native,timeline.source_sha256,timeline.input_format,timeline.duration_seconds,lineage_directory)
    if file_sha256(native_path)!=native['array_sha256']:raise ValueError('native fMRI checksum mismatch')
    segments=native['segments'];starts=np.array([s['start'] for s in segments]);durations=np.array([s['duration'] for s in segments])
    with np.load(native_path,allow_pickle=False) as data:
        values=data['predicted_cortical_response']
        if (values.ndim!=2 or not all(values.shape) or list(values.shape)!=native['shape']
                or len(values)!=len(segments) or not np.isfinite(values).all()):
            raise ValueError('invalid native cortical array')
    # Native TRIBE clocks may start before source zero (the authenticated
    # text/PDF fixtures start at -1 microsecond). Pass those timestamps through;
    # the separate quality display still has nonnegative integer-second queries.
    if (not np.isfinite(starts).all() or not np.isfinite(durations).all()
            or (durations<=0).any() or len({s['timeline'] for s in segments})!=1
            or np.any(starts[1:]<starts[:-1]+durations[:-1]-1e-8)):
        raise ValueError('invalid native cortical clock')
    destination.mkdir(parents=True,exist_ok=False)
    # Byte copy, not resampling or reconstruction of the native array.
    shutil.copyfile(native_path,destination/'fmri.npz')
    shutil.copyfile(manifest_path,destination/'native-tribe-manifest.json')
    if bound['lineage_manifest_sha256']:
        shutil.copyfile(Path(lineage_directory)/'manifest.json',destination/'source-lineage.json')
        if file_sha256(destination/'source-lineage.json') != bound['lineage_manifest_sha256']:
            raise ValueError('lineage changed during export')
    (destination/'timeline.json').write_text(timeline.model_dump_json(indent=2)+'\n')
    if (file_sha256(destination/'fmri.npz')!=native['array_sha256']
            or file_sha256(native_path)!=native['array_sha256']
            or file_sha256(manifest_path)!=manifest_hash
            or file_sha256(destination/'native-tribe-manifest.json')!=manifest_hash):
        raise ValueError('native output changed during export')
    result={'schema':'perception-lesson-bundle-v1','source_sha256':timeline.source_sha256,
            'model_version':timeline.model_version,'artifact_sha256':timeline.artifact_sha256,
            'evaluation_status':evaluation_status,'quality_timeline':{'file':'timeline.json',
                'sha256':file_sha256(destination/'timeline.json'),'query_step_seconds':1,'context_seconds':44,
                'psychological_resolution_validated_seconds':None,
                'supported_points_by_quality':{q:sum(p.qualities[q].value is not None for p in timeline.points) for q in timeline.points[0].qualities}},
            'fmri':{'file':'fmri.npz','sha256':native['array_sha256'],
                'native_manifest':'native-tribe-manifest.json','native_manifest_sha256':manifest_hash,
                'clock':'unchanged native TRIBE segments','checkpoint_revision':native.get('checkpoint_revision'),
                'native_start_seconds':float(starts[0]),
                'native_stop_seconds':float((starts+durations).max()),
                'contains_pre_source_zero_frame':bool((starts<0).any()),
                'spatial_metadata':native.get('spatial_metadata'),'native_equivalence_independently_checked':native.get('native_equivalence_independently_checked',False),
                'TRIBE_weights_changed':False,'observed_fmri':False,'display_interpolation_applied':False},
            'source_binding':{k:v for k,v in bound.items() if k!='schedule'},
            'reference_population':timeline.reference_population,'participant_data_included':False,
            'student_agents_executed':False,'synthetic_label_agreement_is_human_accuracy':False,
            'files_exclude_model_weights':True}
    (destination/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

