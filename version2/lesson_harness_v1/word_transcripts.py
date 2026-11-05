"""Read existing public-content ASR caches without manufacturing word clocks.

Cache location, source identity and original event-count audits are checked.
These checks establish consistency, not independent manual ASR validation.
"""
import csv
import json
import math
from pathlib import Path

from brain.worker_contract import file_sha256
from brain.tribe_events import audit_word_rows


def load_words(directory, lesson, feature_path, feature_digest):
    directory, feature_path = Path(directory), Path(feature_path)
    receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    matches = [r for r in receipt['records'] if r['lesson'] == lesson]
    if len(matches) != 1 or 'error' in matches[0]:
        raise ValueError('one successfully retrieved transcript required')
    record = matches[0]
    if (file_sha256(feature_path) != feature_digest
            or record['feature_manifest_sha256'] != feature_digest):
        raise ValueError('transcript feature-source mismatch')
    feature = json.loads(feature_path.read_text())
    sha = feature['source_sha256']
    if (record['source_sha256'] != sha or record['remote_volume'] != 'perception-public-model-cache'
            or record['remote_path'] != f'public-content/{sha}/input.tsv'):
        raise ValueError('transcript cache identity mismatch')
    path = (directory / record['local_file']).resolve()
    if path.parent != directory.resolve() or file_sha256(path) != record['sha256']:
        raise ValueError('transcript file changed')
    native_pin = feature['source_manifests']['tribe']
    native_path = Path(native_pin['path'])
    if file_sha256(native_path) != native_pin['sha256']:
        raise ValueError('native event audit changed')
    native = json.loads(native_path.read_text())
    if native['source']['sha256'] != sha:
        raise ValueError('native event media mismatch')
    with path.open(newline='') as handle:
        # Sentence fields can contain text beyond a selected interval. Never use
        # them: retain only the actual individual word and its cached clock.
        words = [{'text': r['text'], 'start': float(r['start']), 'duration': float(r['duration'])}
                 for r in csv.DictReader(handle, delimiter='\t')]
    original = native['event_preprocessing_audit']
    audit = audit_word_rows(words, original['media_stop'])
    if (len(words) != original['original_transcript_words']
            or any(b['start'] < a['start'] for a, b in zip(words, words[1:]))):
        raise ValueError('cached transcript count/order mismatch')
    if original['words_removed_by_native_missing_context_filter'] == 0:
        if (audit['word_count'] != original['word_count'] or not math.isclose(
                audit['maximum_word_stop'], original['maximum_word_stop'], abs_tol=1e-6)):
            raise ValueError('cached transcript differs from original event audit')
    spans = [{'id': f'{lesson}-word-{i}', 'start': w['start'], 'stop': w['start'] + w['duration'],
              'text': w['text'], 'timing': 'cached ASR/alignment word interval; not manually validated'}
             for i, w in enumerate(words)]
    pin = {'receipt_sha256': file_sha256(receipt_path), 'transcript_sha256': record['sha256'],
           'native_manifest': native_pin, 'audit': audit, 'asr_independently_validated': False}
    return spans, pin


def pack_contained_words(spans, start, stop):
    """Compact prompt evidence without sentence context outside the window."""
    selected = [s for s in spans if s['start'] >= start and s['stop'] <= stop and s['text'].strip()]
    if not selected:
        return []
    return [{'id': 'context-words', 'start': selected[0]['start'], 'stop': max(s['stop'] for s in selected),
             'text': ' '.join(s['text'] for s in selected),
             'timing': 'only cached ASR words wholly inside context; pauses and gaps are not filled',
             'word_ids': [s['id'] for s in selected]}]
