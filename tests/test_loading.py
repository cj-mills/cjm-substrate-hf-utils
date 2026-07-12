"""Tests for cjm_substrate_hf_utils.loading — from_pretrained with typed OOM.

Projected from the loading notebook's test cell at the c25780e8 flip (model
classes mocked — no real model download)."""
import pytest
import torch

from cjm_substrate.core.errors import CapabilityResourceError
from cjm_substrate_hf_utils.loading import load_pretrained_with_oom


class _FakeModel:
    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        return {"repo_id": repo_id, "kwargs": kwargs}


class _OOMModel:
    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")


class _ValueErrModel:
    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        raise ValueError("bad config")


def test_success_path_forwards_kwargs_verbatim():
    m = load_pretrained_with_oom(_FakeModel, "org/m", device_map="cuda", dtype="bf16")
    assert m["repo_id"] == "org/m"
    assert m["kwargs"]["device_map"] == "cuda" and m["kwargs"]["dtype"] == "bf16"


def test_cuda_oom_converts_with_cause_preserved():
    with pytest.raises(CapabilityResourceError) as ei:
        load_pretrained_with_oom(_OOMModel, "org/m")
    assert ei.value.resource_shortfall is not None
    assert ei.value.resource_shortfall.resource == "gpu_vram_mb"
    assert isinstance(ei.value.__cause__, torch.cuda.OutOfMemoryError)


def test_non_oom_errors_propagate_unchanged():
    with pytest.raises(ValueError):
        load_pretrained_with_oom(_ValueErrModel, "org/m")
