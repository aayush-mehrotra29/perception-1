"""Reuse measured descriptor sources at new queries; never resample predictions."""
from pathlib import Path
import numpy as np

from brain.auxiliary_windows import semantic_windows, sam_audio_windows, sam_visual_windows


def assemble(spec, queries, source_sha256, duration):
    if not isinstance(spec,dict) or spec.get('source_sha256')!=source_sha256 or spec.get('media_duration')!=duration:
        raise ValueError('auxiliary source identity/clock mismatch')
    q=np.asarray(queries,dtype=float)
    if q.ndim!=1 or not np.isfinite(q).all() or (q<0).any() or (q>duration).any() or (np.diff(q)<=0).any():
        raise ValueError('ordered content queries required')
    blocks={'progress':q[:,None]/duration}
    if 'semantics' in spec:
        blocks['semantics'],_=semantic_windows(Path(spec['semantics']),q,source_sha256)
    if 'sam_audio' in spec:
        blocks['sam_audio'],_=sam_audio_windows(Path(spec['sam_audio']),Path(spec['audio_derivative']),q,source_sha256)
    if 'sam_visual' in spec:
        # Rendered documents have an authenticated derivative identity/clock.
        visual_source=spec.get('rendered_visual_source_sha256',source_sha256)
        blocks['sam_visual'],_=sam_visual_windows(Path(spec['sam_visual']),Path(spec['frames']),q,visual_source)
    if 'sensory_directory' in spec:
        from brain.sensory import window_blocks
        measured,meta=window_blocks(Path(spec['sensory_directory']),q,source_sha256)
        if meta['qualification_prefix_only'] or meta['duration_seconds']!=duration:
            raise ValueError('incomplete sensory coverage')
        blocks.update(measured)
    elif 'sensory_data_root' in spec:
        from brain.local_baseline import load_sensory,SOURCES
        from brain.fusion_data import read_json
        measured,observed_duration,_=load_sensory(Path(spec['sensory_data_root']),spec['lecture'],q)
        summary=read_json(Path(spec['sensory_data_root'])/'runs'/SOURCES[spec['lecture']][1]/'summary.json')
        pins=[sha for name,sha in summary['input_pins'].items() if Path(name).suffix.lower()=='.mp4']
        if observed_duration!=duration or pins!=[source_sha256]:raise ValueError('sensory media identity mismatch')
        blocks.update(measured)
    return blocks

