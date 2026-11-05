"""Bounded official ZIP-member acquisition and label eligibility audit.

Experiment 2 only: no reserved content families are read. Retains raw response
codes without pretending that uncertain ordinal semantics are calibrated scores.
Does not import any rating into training or read an EEG recording.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import struct
import urllib.request
import zlib

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256

MEMBER = 'phenotype/stimuli_feedback_meta.tsv'
PROTECTED = {'06', '10', '11'}


def bounded_range(url, start, stop):
    if not url.startswith('https://zenodo.org/api/records/') or not 0 <= start <= stop or stop-start > 65535:
        raise ValueError('bounded official archive range required')
    request = urllib.request.Request(url, headers={
        'Range': f'bytes={start}-{stop}', 'User-Agent': 'Perception-secondary-analysis/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{stop}/'):
            raise ValueError('server did not honor exact range; whole archive refused')
        data = response.read(stop-start+2)
        if len(data) != stop-start+1:
            raise ValueError('incorrect response size')
    return data


def member_bytes(url, member, fetch=bounded_range):
    if member['name'] != MEMBER or not 0 < member['bytes'] <= 65536 or not 0 < member['compressed_bytes'] <= 65536:
        raise ValueError('only bounded feedback member permitted')
    offset = member['offset']
    header = fetch(url, offset, offset+29)
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack('<IHHHHHIIIHH', header)
    if signature != 0x04034b50 or flags & 1 or method not in {0, 8} or name_len < 1:
        raise ValueError('unsupported ZIP local header')
    if fetch(url, offset+30, offset+29+name_len).decode() != MEMBER:
        raise ValueError('ZIP member identity mismatch')
    start = offset+30+name_len+extra_len
    raw = fetch(url, start, start+member['compressed_bytes']-1)
    if method == 8:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(raw, member['bytes']+1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('invalid or oversized deflate member')
    else:
        data = raw
    if len(data) != member['bytes'] or f'{zlib.crc32(data):08x}' != member['CRC32']:
        raise ValueError('ZIP member CRC/length mismatch')
    return data


def audit_rows(data, video_rows, metadata):
    mapping = {r['video_no_local'].zfill(2): r['video_no'].zfill(2)
               for r in video_rows if r['experiment'] == '2'}
    if set(mapping) != {'01', '02', '03', '04', '05'} or set(mapping.values()) & PROTECTED:
        raise ValueError('experiment 2 mapping incomplete or protected')
    if metadata['questions']['engaging'] != 'I found this video engaging':
        raise ValueError('engagement question changed')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t')
    if set(reader.fieldnames or []) != {'participant_id', 'stimulus', 'engaging', 'enjoyable', 'educational'}:
        raise ValueError('unexpected feedback schema')
    seen = set(); counts = {v: 0 for v in mapping.values()}; people = set(); missing = 0
    for row in reader:
        stimulus = row['stimulus']
        if not stimulus.startswith('Stim-') or stimulus[5:] not in mapping:
            raise ValueError('unknown stimulus; no guessed family mapping')
        key = row['participant_id'], stimulus
        if key in seen or not row['participant_id'].startswith('sub-'):
            raise ValueError('duplicate or invalid subject stimulus pair')
        seen.add(key); people.add(row['participant_id'])
        code = row['engaging'].strip()
        if code in {'', 'n/a', 'NA', 'NaN'}:
            missing += 1; continue
        if code not in {str(i) for i in range(1, 8)}:
            raise ValueError('unsupported engagement code')
        counts[mapping[stimulus[5:]]] += 1
    return {'experiment': 2, 'total_rows': len(seen), 'participants': len(people),
        'engagement_ratings_by_family': counts, 'nonmissing_engagement_ratings': sum(counts.values()),
        'missing_engagement_ratings': missing, 'label_scope': 'whole_video',
        'question': metadata['questions']['engaging'], 'published_answer_codes': metadata['answers'],
        'ordinal_mapping_verified': False, 'model_training_eligible': False,
        'unresolved': ['Published answer ordering is nonmonotonic in agreement strength; verify response-code semantics against original experiment code.',
            'Attentive/distracted session pairing and repeated-viewing scope need source confirmation.',
            'Media identity, presentation alignment and matching frozen representations are not qualified.',
            'Subject identities across experiments must be audited before participant-disjoint splitting.'],
        'protected_families_read': [], 'EEG_payloads_read': False, 'training_performed': False,
        'educational_self_report_is_quiz_comprehension': False}


def acquire(review, output):
    review, output = map(Path, (review, output)); guard(output)
    if output.exists():
        raise ValueError('preserve existing audits')
    directory = review/'archive-review-v2/experiment2'
    summary = json.loads((directory/'summary.json').read_text())
    members = json.loads((directory/'members.json').read_text())
    member = next(m for m in members if m['name'] == MEMBER)
    metadata = json.loads((directory/'phenotype/stimuli_feedback_meta.json').read_text())
    with (review/'videos_by_experiment.csv').open() as stream:
        videos = list(csv.DictReader(stream))
    # Qualify protection and schema before the HTTP request.
    audit_rows(b'participant_id\tstimulus\tengaging\tenjoyable\teducational\n', videos, metadata)
    data = member_bytes(summary['archive_url'], member)
    report = audit_rows(data, videos, metadata)
    guard(output); output.mkdir(parents=True)
    (output/'experiment2_feedback.tsv').write_bytes(data)
    report.update({'source_url': summary['archive_url'], 'archive_member': member,
        'source_manifest_sha256': file_sha256(directory/'summary.json'),
        'source_member_inventory_sha256': file_sha256(directory/'members.json'),
        'source_sidecar_sha256': file_sha256(directory/'phenotype/stimuli_feedback_meta.json'),
        'source_video_mapping_sha256': file_sha256(review/'videos_by_experiment.csv'),
        'download_sha256': file_sha256(output/'experiment2_feedback.tsv'),
        'whole_archive_hash_verified': False, 'member_CRC_verified': True,
        'code_sha256': file_sha256(Path(__file__))})
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('review', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); r = acquire(args.review, args.output)
    print(json.dumps({k: r[k] for k in ('total_rows', 'participants', 'nonmissing_engagement_ratings', 'model_training_eligible')}))
