"""Bind upload clocks to native inference derivatives without inventing reading time."""
import json
from pathlib import Path

from brain.worker_contract import file_sha256
from .contract import SourceReference


def binding(native, source_sha256, input_format, duration, lineage_directory=None):
    if input_format not in {'video','audio','text','pdf','image'}:
        raise ValueError('unsupported input format')
    original = native.get('source', {})
    schedule = native.get('assumed_presentation_schedule') or {}
    proof = None
    if input_format in {'video','text'} and original.get('sha256') != source_sha256:
        raise ValueError('quality/TRIBE source mismatch')
    if input_format in {'pdf','image'} or original.get('sha256') != source_sha256:
        if lineage_directory is None:
            raise ValueError('authenticated source derivative lineage required')
        path = Path(lineage_directory) / 'manifest.json'
        proof = json.loads(path.read_text())
        if (proof.get('parent',{}).get('sha256') != source_sha256
                or proof.get('parent',{}).get('media_kind') != input_format
                or proof.get('derivative') != original):
            raise ValueError('upload/native source lineage mismatch')
        if input_format in {'pdf','image'}:
            if (proof.get('schema') != 'perception-document-presentation-v1'
                    or proof.get('schedule') != schedule
                    or file_sha256(Path(lineage_directory)/'presentation.mp4') != original['sha256']):
                raise ValueError('document presentation identity mismatch')
        elif input_format == 'audio':
            if (proof.get('schema') != 'perception-audio-derivative-v1'
                    or not proof.get('audio_clock_continuity_checked')
                    or abs(proof['samples']/proof['sample_rate']-duration)>1e-6
                    or file_sha256(Path(lineage_directory)/'audio.wav') != original['sha256']):
                raise ValueError('audio derivative identity/clock mismatch')
        else:
            raise ValueError('unsupported derivative mapping')
    elif original.get('media_kind') != input_format:
        raise ValueError('native input format mismatch')
    if input_format in {'pdf','image','text'}:
        key = 'source_text_utf8_sha256' if input_format == 'text' else 'source_sha256'
        if (schedule.get('input_format') != input_format or schedule.get(key) != source_sha256
                or abs(schedule.get('duration_seconds',-1)-duration)>1e-6):
            raise ValueError('assumed presentation/source/clock mismatch')
        basis = schedule['clock_basis']
    else:
        basis = 'native media time; decoded audio derivative when documented'
    return {'source_sha256':source_sha256,'native_source_sha256':original['sha256'],
            'input_format':input_format,'duration_seconds':duration,'clock_basis':basis,
            'schedule':schedule,'lineage_manifest_sha256':file_sha256(Path(lineage_directory)/'manifest.json') if proof else None}


def references(source, second):
    """References describe the full preceding context, not an inferred gaze position."""
    if second < 44:return []
    start, stop = second-44, second
    kind, digest = source['input_format'], source['source_sha256']
    if kind in {'video','audio'}:
        return [SourceReference(source_id=digest,kind='media_interval',start=start,stop=stop)]
    schedule = source['schedule']
    if kind == 'text':
        words = [w for w in schedule['words'] if w['start'] < stop and w['start']+w['duration'] > start]
        if not words:return []
        return [SourceReference(source_id=digest,kind='text_span',character_start=words[0]['character_start'],character_stop=words[-1]['character_stop'])]
    pages = [p for p in schedule['pages'] if p['start'] < stop and p['start']+p['duration'] > start]
    return [SourceReference(source_id=digest,kind='page' if kind=='pdf' else 'image',
                            page_index=p['source_page_index'] if kind=='pdf' else None) for p in pages]

