"""Call model_class.from_pretrained(...) and convert CUDA OOM into the substrate's typed CapabilityResourceError for CR-7 reactive retry.

Thin wrapper over the `from_pretrained` convention (transformers, qwen_asr, ...):
forwards every kwarg and converts ONLY CUDA OOM — `is_cuda_oom`: the caching-allocator
subclass or a driver-level RuntimeError (via cjm-substrate-torch-utils' primitives);
every other exception propagates unchanged for the substrate's CR-5 classifier. Consumers whose load is a plain constructor
(e.g. LavaSR's `LavaEnhance2(path)`) should call
`cuda_oom_to_capability_resource_error` directly instead."""

from typing import Any, Optional, Type

from cjm_substrate_torch_utils.oom import cuda_oom_to_capability_resource_error, is_cuda_oom


def load_pretrained_with_oom(
    model_class: Type,            # A class exposing `.from_pretrained(repo_id, **kwargs)`
    repo_id: str,                 # Model repo id or local path passed to from_pretrained
    *,
    label: Optional[str] = None,  # OOM-message context (default: f"loading {repo_id!r}")
    **kwargs,                     # Forwarded verbatim to model_class.from_pretrained
) -> Any:                         # The loaded model
    """Call `model_class.from_pretrained(repo_id, **kwargs)`, converting CUDA OOM to a typed error.

    On any CUDA OOM (`is_cuda_oom`: the caching-allocator subclass OR a driver-level
    RuntimeError), re-raises as `CapabilityResourceError` (via
    `cuda_oom_to_capability_resource_error`) so the substrate's CR-7 reactive-retry
    path can evict + reload + retry. All other exceptions propagate unchanged.

    Wrap the call in `self.heartbeat(...)` at the capability site to cover the silent
    construction phase; pre-download with `snapshot_download_with_progress` for
    real download progress.
    """
    try:
        return model_class.from_pretrained(repo_id, **kwargs)
    except Exception as e:
        # is_cuda_oom: caching-allocator subclass OR driver-level RuntimeError OOM;
        # everything else propagates unchanged for the substrate's CR-5 classifier.
        if not is_cuda_oom(e):
            raise
        raise cuda_oom_to_capability_resource_error(
            e, label=label or f"loading {repo_id!r}",
        ) from e
