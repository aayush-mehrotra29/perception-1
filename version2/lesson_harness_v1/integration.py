"""Attach the lesson harness to completed content artifacts without new paid calls."""
import json
from pathlib import Path

from brain.worker_contract import file_sha256
from .concepts import collect
from .inference import predict_native


def attach(output,request,model,content_result):
    output=Path(output)
    native=json.loads((output/'tribe/manifest.json').read_text())
    kind=request.media_kind
    if kind=='video':
        spec=json.loads((output/'feature-sources.json').read_text())
        duration=spec['media_duration'];lineage=None
    else:
        lineage=output/'audio' if kind=='audio' else output/'presentation' if kind in {'pdf','image'} else None
        if kind=='audio':
            audio=json.loads((lineage/'manifest.json').read_text())
            duration=audio['samples']/audio['sample_rate']
        else:duration=native['assumed_presentation_schedule']['duration_seconds']
        spec={'source_sha256':request.sha256,'media_duration':duration}
        if (output/'semantics/manifest.json').is_file():spec['semantics']=str((output/'semantics').resolve())
        if (output/'sam_audio/manifest.json').is_file():
            spec.update(sam_audio=str((output/'sam_audio').resolve()),audio_derivative=str((output/'audio').resolve()))
        if (output/'sam_visual/manifest.json').is_file():
            spec.update(sam_visual=str((output/'sam_visual').resolve()),frames=str((output/'frames').resolve()),
                        rendered_visual_source_sha256=native['source']['sha256'])
    concepts=collect(Path(spec['semantics']),request.sha256) if 'semantics' in spec else []
    bundle=predict_native(output/'tribe',model,output/'lesson-prediction',duration=duration,input_format=kind,
        source_sha256=request.sha256,lineage_directory=lineage,feature_sources=spec,concept_occurrences=concepts)
    (output/'content-manifest.json').write_text(json.dumps(content_result,indent=2,allow_nan=False)+'\n')
    updated={**content_result,'lesson_prediction':{'bundle':'lesson-prediction',
        'manifest_sha256':file_sha256(output/'lesson-prediction/manifest.json'),
        'model_version':bundle['model_version'],'artifact_sha256':bundle['artifact_sha256'],
        'supported_points_by_quality':bundle['quality_timeline']['supported_points_by_quality'],
        'evaluation_status':bundle['evaluation_status']},
        'attention_predictions_included':bundle['quality_timeline']['supported_points_by_quality']['attention_difficulty']>0,
        'retention_predictions_included':False,'student_states_observed':False,
        'quality_estimates_included':True,'content_manifest_sha256':file_sha256(output/'content-manifest.json')}
    # Retain the old generic flag only where that schema already contains it.
    if 'attention_or_retention_predictions' in updated:
        updated['attention_or_retention_predictions']=updated['attention_predictions_included']
    (output/'manifest.json').write_text(json.dumps(updated,indent=2,allow_nan=False)+'\n')
    return updated
