import struct
import zlib

import pytest

from lesson_harness_v1.bbbd_engagement_audit import MEMBER, audit_rows, member_bytes


def metadata():
    return {'questions': {'engaging': 'I found this video engaging'}, 'answers': {'ans1': 'Strongly Disagree'}}


def videos():
    return [{'experiment': '2', 'video_no_local': f'{i:02}', 'video_no': f'{i:02}'} for i in range(1, 6)]


def test_labels_keep_whole_video_scope_and_no_guessed_scale():
    rows = b'participant_id\tstimulus\tengaging\tenjoyable\teducational\nsub-01\tStim-01\t6\t7\t7\nsub-02\tStim-02\tn/a\t5\t5\n'
    report = audit_rows(rows, videos(), metadata())
    assert report['nonmissing_engagement_ratings'] == 1
    assert report['missing_engagement_ratings'] == 1
    assert report['label_scope'] == 'whole_video'
    assert not report['model_training_eligible'] and not report['ordinal_mapping_verified']


def test_reserved_family_mapping_rejected_before_reading():
    mapping = videos(); mapping[0]['video_no'] = '06'
    with pytest.raises(ValueError, match='protected'):
        audit_rows(b'', mapping, metadata())


def test_duplicate_and_invalid_codes_rejected():
    header = b'participant_id\tstimulus\tengaging\tenjoyable\teducational\n'
    row = b'sub-01\tStim-01\t6\t7\t7\n'
    with pytest.raises(ValueError, match='duplicate'):
        audit_rows(header+row+row, videos(), metadata())
    with pytest.raises(ValueError, match='code'):
        audit_rows(header+row.replace(b'\t6\t', b'\t99\t'), videos(), metadata())


@pytest.mark.parametrize('compressed', [False, True])
def test_member_crc_and_identity(compressed):
    data = b'tiny fixture'
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(data)+encoder.flush() if compressed else data
    name = MEMBER.encode()
    crc = zlib.crc32(data)
    header = struct.pack('<IHHHHHIIIHH', 0x04034b50, 20, 0, 8 if compressed else 0, 0, 0, crc, len(payload), len(data), len(name), 0)
    archive = header+name+payload
    rec = {'name': MEMBER, 'bytes': len(data), 'compressed_bytes': len(payload), 'offset': 0, 'CRC32': f'{crc:08x}'}
    fetch = lambda url, start, stop: archive[start:stop+1]
    assert member_bytes('fixture', rec, fetch) == data
    rec['CRC32'] = '00000000'
    with pytest.raises(ValueError, match='CRC'):
        member_bytes('fixture', rec, fetch)
