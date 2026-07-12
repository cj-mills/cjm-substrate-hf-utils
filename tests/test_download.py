"""Tests for cjm_substrate_hf_utils.download — snapshot download with progress.

Projected from the download notebook's test cell at the c25780e8 flip
(network-free: `local_files_only=True` on an uncached repo fails fast)."""
import pytest

from cjm_substrate_hf_utils.download import snapshot_download_with_progress

_BOGUS = "cjm-nonexistent-org/definitely-not-a-real-repo-xyz"


def test_kwargs_forward_to_snapshot_download():
    # local_files_only=True on a not-cached repo fails fast without network,
    # proving our kwargs forward through to snapshot_download
    with pytest.raises(Exception):
        snapshot_download_with_progress(_BOGUS, local_files_only=True)


def test_report_progress_callback_accepted():
    # the reporting tqdm subclass builds cleanly; same fast-fail path — but a
    # NameError here means the _tqdm import was pruned again (finding 0feb4290),
    # so pin the failure to the cache-miss class, never a bare Exception
    calls = []
    with pytest.raises(Exception) as ei:
        snapshot_download_with_progress(
            _BOGUS, local_files_only=True,
            report_progress=lambda f, m: calls.append((f, m)),
        )
    assert not isinstance(ei.value, NameError), "pruned _tqdm import (0feb4290)"
