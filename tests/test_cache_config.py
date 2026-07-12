"""Tests for cjm_substrate_hf_utils.cache_config — the HF cache config mixin.

Projected from the cache_config notebook's test cell at the c25780e8 flip."""
import dataclasses
from dataclasses import field

from cjm_substrate.core.capability import RELOAD_TRIGGER
from cjm_substrate.utils.validation import (SCHEMA_TITLE, config_to_dict,
                                            dataclass_to_jsonschema, dict_to_config)
from cjm_substrate_hf_utils.cache_config import HFCacheConfig


@dataclasses.dataclass
class _DemoConfig(HFCacheConfig):
    model_id: str = field(default="org/m", metadata={RELOAD_TRIGGER: "model", SCHEMA_TITLE: "Model ID"})


def test_mixin_fields_present_defaulted_and_reload_tagged():
    fields = {f.name: f for f in dataclasses.fields(_DemoConfig)}
    for n in ("cache_dir", "revision", "local_files_only", "trust_remote_code", "model_id"):
        assert n in fields, n
    for n in ("cache_dir", "revision", "local_files_only", "trust_remote_code"):
        assert fields[n].metadata[RELOAD_TRIGGER] == "model"


def test_defaults_are_safe():
    c = _DemoConfig()
    assert c.cache_dir is None and c.revision is None
    assert c.local_files_only is False and c.trust_remote_code is False


def test_round_trips_through_substrate_config_helpers():
    d = config_to_dict(_DemoConfig(cache_dir="/tmp/hf", trust_remote_code=True))
    assert d["cache_dir"] == "/tmp/hf" and d["trust_remote_code"] is True
    c2 = dict_to_config(_DemoConfig, {"revision": "abc123", "model_id": "org/y"})
    assert c2.revision == "abc123" and c2.model_id == "org/y"


def test_jsonschema_titles_for_config_ui():
    schema = dataclass_to_jsonschema(_DemoConfig)
    assert "cache_dir" in schema["properties"]
    assert schema["properties"]["cache_dir"]["title"] == "HF Cache Directory"
