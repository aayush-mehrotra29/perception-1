"""Neutral argument bridge for SAM's legacy ModelHubMixin implementation."""
from functools import wraps


def legacy_hub_defaults(original):
    @wraps(original)
    def compatible(cls, **kwargs):
        # Hub 1.x removes these deprecated arguments before calling the model's
        # loader. SAM still requires them. Restore only their neutral defaults.
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return original(cls, **kwargs)
    return compatible


def install():
    from sam_audio.model.base import BaseModel
    if not getattr(BaseModel, "_perception_hub_compat", False):
        BaseModel._from_pretrained = classmethod(legacy_hub_defaults(BaseModel._from_pretrained.__func__))
        BaseModel._perception_hub_compat = True
