"""A dataclass mixin adding HuggingFace Hub cache / revision / air-gap / security fields to a capability's config, with RELOAD_TRIGGER + JSON-schema metadata pre-set.

Capabilities that load models from the HuggingFace Hub compose `HFCacheConfig` into
their config dataclass by inheritance. All four fields are `RELOAD_TRIGGER`-tagged
(`"model"`): changing any of them invalidates a loaded model, so the substrate's CR-4
reconfigure path fires `_release_model` before re-applying config. Dataclass
inheritance places these (defaulted) fields FIRST in `__init__` order, so the
consuming capability's own fields must also carry defaults."""

from dataclasses import dataclass, field
from typing import Optional

from cjm_substrate.core.capability import RELOAD_TRIGGER
from cjm_substrate.utils.validation import SCHEMA_DESC, SCHEMA_TITLE


@dataclass
class HFCacheConfig:
    """Mixin adding HuggingFace Hub cache/revision/air-gap/security fields to a capability config.

    Compose by inheritance:

        @dataclass
        class MyCapabilityConfig(HFCacheConfig):
            model_id: str = field(default="org/model", metadata={RELOAD_TRIGGER: "model"})
            # ... capability-specific fields (all defaulted) ...

    Each field is `RELOAD_TRIGGER`-tagged `"model"` (a change invalidates a loaded
    model) and carries `SCHEMA_TITLE`/`SCHEMA_DESC` so the capability-config UI renders
    it. `trust_remote_code` defaults to False and is flagged DANGER in its help text.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            RELOAD_TRIGGER: "model",
            SCHEMA_TITLE: "HF Cache Directory",
            SCHEMA_DESC: "Override the HF Hub cache location for model downloads (default: HF_HOME).",
        },
    )
    revision: Optional[str] = field(
        default=None,
        metadata={
            RELOAD_TRIGGER: "model",
            SCHEMA_TITLE: "Model Revision",
            SCHEMA_DESC: "Git revision / tag / commit to pin (default: the repo's main branch).",
        },
    )
    local_files_only: bool = field(
        default=False,
        metadata={
            RELOAD_TRIGGER: "model",
            SCHEMA_TITLE: "Local Files Only",
            SCHEMA_DESC: "Reject network access and use only cached files (air-gapped / reproducible).",
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            RELOAD_TRIGGER: "model",
            SCHEMA_TITLE: "Trust Remote Code",
            SCHEMA_DESC: "DANGER: allow execution of model-author code shipped in the repo.",
        },
    )
