import numpy as np

from brain.evaluate import evaluate_crossed
from brain.eeg import EEGModel


def test_real_fits_with_held_target_perturbation_do_not_change_forecasts(tmp_path):
    rng = np.random.default_rng(11)
    # Two lectures per family, four content windows per lecture, 16 people.
    rows = [(f, l, p, w) for f in range(2) for l in range(2) for p in range(1, 17) for w in range(4)]
    content = rng.normal(size=(16, 10))
    index = np.array([f * 8 + l * 4 + w for f, l, p, w in rows])
    x = content[index]
    y = x[:, :8] + rng.normal(scale=.1, size=(len(rows), 8))
    ids = np.array([str(i) for i in index])
    lessons = np.array([f"{f}-{l}" for f, l, p, w in rows])
    families = np.array([("aerospace", "xray")[f] for f, l, p, w in rows])
    people = np.array([p for f, l, p, w in rows])
    first = evaluate_crossed({"dense": x}, y, ids, lessons, families, people, tmp_path / "first")
    changed = y.copy()
    changed[(families == "aerospace") & (people <= 4)] += 10000
    second = evaluate_crossed({"dense": x}, changed, ids, lessons, families, people, tmp_path / "second")
    assert len(first["cells"]) == 8 and all(c["status"] == "scored" for c in first["cells"])
    with np.load(tmp_path / "first/aerospace-group1/forecasts.npz") as a, np.load(tmp_path / "second/aerospace-group1/forecasts.npz") as b:
        np.testing.assert_array_equal(a["predictions"], b["predictions"])
    assert first["cells"][0]["mae"] != second["cells"][0]["mae"]
    assert not set(EEGModel.load(tmp_path / "first/aerospace-group1").fit_content_ids) & set(ids[(families == "aerospace")])
