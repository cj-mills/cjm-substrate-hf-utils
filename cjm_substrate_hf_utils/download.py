"""Wrap huggingface_hub.snapshot_download so per-file progress flows to the capability's report_progress, defeating the substrate's stall detector during downloads.

The download phase is where HF Hub capabilities spend their cold-start time, silent
enough to trip the prefetch stall detector. Wrap BOTH this download AND the subsequent
`from_pretrained(..., local_files_only=True)` in one `with self.heartbeat(...)` block:
the per-file tqdm callback is real progress but not a reliable stall-detector floor on
its own (it can be sparse, and one large file reports no advance for a long stretch) —
the heartbeat is the floor and this helper's download % layers on top. Returns the
local snapshot directory; constructors that take a path (LavaSR-style) can pass it to
bypass their internal progress-less download."""

from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm as _tqdm

# The nested _ReportingTqdm subclass is the only consumer of _tqdm, and
# block-nested references are invisible to the canonical emit's ref collection
# (finding 0feb4290) — this module-level reference keeps the import alive.
_TQDM_BASE = _tqdm


def snapshot_download_with_progress(
    repo_id: str,                                                    # HF Hub repo id, e.g. "mistralai/Voxtral-Mini-3B-2507"
    *,
    report_progress: Optional[Callable[[float, str], None]] = None,  # Capability's report_progress(fraction, message)
    cache_dir: Optional[str] = None,                                 # HF cache override (HFCacheConfig.cache_dir)
    revision: Optional[str] = None,                                  # Git rev / tag / commit to pin
    local_files_only: bool = False,                                  # Use cached files only; no network
    **kwargs,                                                        # Forwarded to huggingface_hub.snapshot_download
) -> Path:                                                           # Local path to the downloaded snapshot
    """Download an HF Hub snapshot, streaming per-file progress to `report_progress`.

    When `report_progress` is given, a `tqdm_class` subclass forwards each update
    as `report_progress(downloaded / total, "downloading <file>")`. When it is
    None, the default HF Hub progress bars are used unchanged.

    Returns the local snapshot directory; subsequent `from_pretrained` calls with
    the same `cache_dir` + `local_files_only=True` hit the populated cache.

    Adoption: the per-file tqdm callback is real progress but NOT a reliable
    stall-detector floor on its own (it can be sparse / silent on one large file).
    Wrap BOTH this call and the subsequent `from_pretrained` in a single
    `with self.heartbeat(...)` block so the substrate always sees the tuple advance.
    """
    tqdm_class = None
    if report_progress is not None:
        class _ReportingTqdm(_tqdm):
            def update(self, n=1):
                super().update(n)
                if self.total:
                    report_progress(self.n / self.total, f"downloading {self.desc or repo_id}")
        tqdm_class = _ReportingTqdm
    return Path(snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
        tqdm_class=tqdm_class,
        **kwargs,
    ))
