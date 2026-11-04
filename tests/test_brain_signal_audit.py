import numpy as np

from brain.signal_audit import scan


def test_signal_audit_preserves_receiver_bursts_and_reports_missing_values(tmp_path):
    path = tmp_path / "eeg.csv"
    path.write_text("timestamp,tp9,af7,af8,tp10\n10,1,2,3,4\n10,1,4,5,6\n9,nan,6,7,8\n")
    result = scan(path)
    assert result["rows"] == 3
    assert result["nonfinite_samples"] == [1, 0, 0, 0]
    assert result["equal_neighbor_receiver_timestamps"] == 1
    assert result["decreasing_receiver_timestamps"] == 1
    np.testing.assert_allclose(result["means"], [1, 4, 5, 6])
    np.testing.assert_allclose(result["population_standard_deviations"], [0, np.sqrt(8/3), np.sqrt(8/3), np.sqrt(8/3)])
