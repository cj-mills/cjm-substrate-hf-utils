# cjm-substrate-hf-utils

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

Hugging Face Hub helpers for cjm-substrate capabilities: cache-config mixin, progress-reporting snapshot downloads, and typed-OOM model loading.

## Modules

- **`cjm_substrate_hf_utils.cache_config`** — A dataclass mixin adding HuggingFace Hub cache / revision / air-gap / security fields to a capability's config, with RELOAD_TRIGGER + JSON-schema metadata pre-set.
- **`cjm_substrate_hf_utils.download`** — Wrap huggingface_hub.snapshot_download so per-file progress flows to the capability's report_progress, defeating the substrate's stall detector during downloads.
- **`cjm_substrate_hf_utils.loading`** — Call model_class.from_pretrained(...) and convert CUDA OOM into the substrate's typed CapabilityResourceError for CR-7 reactive retry.

## API

### `cjm_substrate_hf_utils.cache_config`

- `HFCacheConfig` _class_ — Mixin adding HuggingFace Hub cache/revision/air-gap/security fields to a capability config.

### `cjm_substrate_hf_utils.download`

- `snapshot_download_with_progress` _function_ — Download an HF Hub snapshot, streaming per-file progress to `report_progress`.

### `cjm_substrate_hf_utils.loading`

- `load_pretrained_with_oom` _function_ — Call `model_class.from_pretrained(repo_id, **kwargs)`, converting CUDA OOM to a typed error.
