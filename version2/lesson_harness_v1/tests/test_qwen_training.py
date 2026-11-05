from lesson_harness_v1 import qwen_training, qwen_benchmark, qwen_deployment
from . import test_deployment, test_mixed_benchmark


def test_qwen_fit_reloads_same_predictions(tmp_path,monkeypatch):
    monkeypatch.setattr(test_deployment,'training',qwen_training)
    test_deployment.test_small_fitted_software_fixture_reproduces_after_reload(tmp_path,monkeypatch)


def test_qwen_benchmark_keeps_common_heldout_rows_and_all_seeds(tmp_path,monkeypatch):
    monkeypatch.setattr(test_mixed_benchmark,'runner',qwen_benchmark)
    test_mixed_benchmark.test_all_arms_use_common_rows_and_scoring_keeps_missing_heads(tmp_path,monkeypatch)


def test_qwen_untrained_heads_stay_blocked(monkeypatch):
    monkeypatch.setattr(test_deployment,'head_status',qwen_deployment.head_status)
    test_deployment.test_untrained_heads_are_blocked_even_when_network_contains_parameters()
