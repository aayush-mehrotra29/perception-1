"""Lightweight release safeguards; no remote calls or participant data."""
import io
import unittest
import numpy as np
from numerical_brain_v1.cortical_replay import (
    arrays_bytes, digest, compare_fixture, _load_dense, _validate_manifest,
    CHECKPOINT_SHA, CONFIG_SHA, REVISION, ENCODERS, PREPROCESSING,
)
from numerical_brain_v1.release_audit import audit_eeg, CHANNELS


class ReplayAuditTests(unittest.TestCase):
    def test_strict_content_geometry_and_no_extra_fields(self):
        arrays = {'text': np.zeros((1, 2, 3072, 200), np.float32),
                  'audio': np.zeros((1, 2, 1024, 200), np.float32)}
        self.assertEqual(set(_load_dense(arrays_bytes(**arrays))), {'text', 'audio'})
        with self.assertRaises(ValueError): _load_dense(arrays_bytes(**arrays, eeg=np.zeros(1)))
        arrays['audio'] = arrays['audio'].astype(np.float64)
        with self.assertRaises(ValueError): _load_dense(arrays_bytes(**arrays))

    def test_parity_checks_hidden_and_selected_cortex(self):
        hidden = np.zeros((1, 200, 1152), np.float32)
        cortex = np.zeros((1, 20484, 100), np.float32)
        q = {'blob': arrays_bytes(hidden=hidden, cortex=cortex[0].T[:2]),
             'segments': [{'start': i, 'duration': 1., 'timeline': 'default'} for i in range(2)],
             'hidden_source_sha256': 'a' * 64, 'cortical_source_sha256': 'b' * 64}
        r = compare_fixture(hidden, cortex, q)
        self.assertTrue(r['passed'] and r['hidden_bitwise_equal'] and r['cortex_bitwise_equal'])
        hidden[0, 0, 0] = 0.01
        with self.assertRaises(AssertionError): compare_fixture(hidden, cortex, q)
        hidden.fill(0); cortex[0, 0, 1] = 0.01
        with self.assertRaises(AssertionError): compare_fixture(hidden, cortex, q)

    def test_duplicate_and_out_of_bounds_cortex_rows_rejected(self):
        hidden = np.zeros((1, 200, 1152), np.float32); cortex = np.zeros((1, 20484, 100), np.float32)
        for starts in [[0, 0], [100], [-1], [1, 0]]:
            q = {'blob': arrays_bytes(hidden=hidden, cortex=np.zeros((len(starts), 20484), np.float32)),
                 'segments': [{'start': i, 'duration': 1., 'timeline': 'default'} for i in starts]}
            with self.assertRaises(ValueError): compare_fixture(hidden, cortex, q)

    def test_encoder_revision_and_preprocessing_are_fail_closed(self):
        m = {'checkpoint_revision': REVISION, 'checkpoint_sha256': CHECKPOINT_SHA,
             'upstream_config_sha256': CONFIG_SHA, 'torch_version': '2.6.0+cu124',
             'content_encoder_revisions_verified': True, 'capture_equivalence_verified': True,
             'capture_max_absolute_error': 0, 'observed_encoder_revisions': dict(ENCODERS),
             'event_preprocessing_audit': {'preprocessing': PREPROCESSING, 'no_duplicate_words': True,
                 'words_within_media_extent': True, 'checkpoint_or_feature_extractors_modified': False, 'media_stop': 57.5},
             'source': {'media_kind': 'audio', 'source_origin': 'public_url', 'cloud_use_basis': 'authorized fixture'}}
        _validate_manifest(m)
        m['observed_encoder_revisions']['facebook/w2v-bert-2.0'] = 'wrong'
        with self.assertRaises(ValueError): _validate_manifest(m)

    def test_numerical_export_preserves_masks_eof_and_source_channel_ids(self):
        q = np.array([0., .02, .04]); mask = np.array([0, 1, 0], dtype=np.uint8)
        values = np.full((3, 40), np.nan); values[1] = 1e-6
        self.assertEqual(audit_eeg(values, q, mask, CHANNELS, .04)['supported_samples'], 1)
        with self.assertRaises(ValueError): audit_eeg(values, q, mask, list(reversed(CHANNELS)), .04)
        with self.assertRaises(ValueError): audit_eeg(values, q, mask, CHANNELS, .02)
        values[0] = 0
        with self.assertRaises(ValueError): audit_eeg(values, q, mask, CHANNELS, .04)


if __name__ == '__main__': unittest.main()
