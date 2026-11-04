from brain.sam_compat import legacy_hub_defaults


def test_hub_bridge_restores_only_missing_legacy_defaults():
    seen = []
    def original(cls, *, model_id, proxies, resume_download, token):
        seen.append((cls, model_id, proxies, resume_download, token))
        return "unchanged_loader_result"
    call = legacy_hub_defaults(original)
    assert call(str, model_id="local-snapshot", token=False) == "unchanged_loader_result"
    assert seen[-1] == (str, "local-snapshot", None, False, False)
    call(str, model_id="local-snapshot", proxies={"test": "value"}, resume_download=True, token=False)
    assert seen[-1][2:4] == ({"test": "value"}, True)
