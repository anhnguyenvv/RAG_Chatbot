"""Tests for Data/pipeline/config.py."""

from __future__ import annotations

import pytest

from pipeline.config import PipelineConfig, _to_bool, _to_csv_list


class TestToBool:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "y", "on", " on "])
    def test_truthy(self, value):
        assert _to_bool(value, default=False) is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "n", "off"])
    def test_falsy(self, value):
        assert _to_bool(value, default=True) is False

    def test_none_returns_default(self):
        assert _to_bool(None, default=True) is True
        assert _to_bool(None, default=False) is False

    def test_unknown_returns_default(self):
        assert _to_bool("maybe", default=True) is True
        assert _to_bool("foo", default=False) is False


class TestToCsvList:
    def test_none_returns_default_copy(self):
        default = ["a", "b"]
        result = _to_csv_list(None, default)
        assert result == ["a", "b"]
        assert result is not default  # copy, not reference

    def test_splits_on_comma(self):
        assert _to_csv_list("a,b,c", []) == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert _to_csv_list("  a , b ,  c  ", []) == ["a", "b", "c"]

    def test_drops_empty_items(self):
        assert _to_csv_list("a,,b,", []) == ["a", "b"]

    def test_empty_string_returns_empty_list(self):
        assert _to_csv_list("", []) == []

    def test_whitespace_only_returns_empty(self):
        assert _to_csv_list("   ", []) == []


class TestPipelineConfigValidate:
    def _valid_config(self, **overrides) -> PipelineConfig:
        base = PipelineConfig(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="key",
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    def test_happy_path(self):
        self._valid_config().validate()  # should not raise

    def test_missing_qdrant_url(self):
        cfg = self._valid_config(qdrant_url="")
        with pytest.raises(ValueError, match="QDRANT_URL"):
            cfg.validate()

    def test_missing_qdrant_key(self):
        cfg = self._valid_config(qdrant_api_key="")
        with pytest.raises(ValueError, match="QDRANT_API_KEY"):
            cfg.validate()

    def test_invalid_chunk_strategy(self):
        cfg = self._valid_config(chunk_strategy="semantic")
        with pytest.raises(ValueError, match="CHUNK_STRATEGY"):
            cfg.validate()

    @pytest.mark.parametrize("strategy", ["recursive", "outline"])
    def test_valid_chunk_strategies(self, strategy):
        self._valid_config(chunk_strategy=strategy).validate()


class TestPipelineConfigFromEnv:
    def test_defaults_when_env_empty(self, monkeypatch):
        # Clear all pipeline-related env vars
        for key in [
            "PIPELINE_SOURCE_DIR", "QDRANT_COLLECTION_NAME", "EMBEDDING_MODEL_NAME",
            "CHUNK_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP", "QDRANT_URL",
            "QDRANT_API_KEY", "QDRANT_FORCE_RECREATE", "HUGGINGFACE_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg = PipelineConfig.from_env()
        assert cfg.source_dir == "./Database"
        assert cfg.collection_name == "ITUS_mpnet_600v1"
        assert cfg.chunk_strategy == "outline"
        assert cfg.chunk_size == 600
        assert cfg.chunk_overlap == 200
        assert cfg.qdrant_force_recreate is True

    def test_overrides_via_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://custom:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "secret")
        monkeypatch.setenv("CHUNK_SIZE", "800")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("CHUNK_STRATEGY", "recursive")
        monkeypatch.setenv("QDRANT_FORCE_RECREATE", "false")

        cfg = PipelineConfig.from_env()
        assert cfg.qdrant_url == "http://custom:6333"
        assert cfg.qdrant_api_key == "secret"
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 100
        assert cfg.chunk_strategy == "recursive"
        assert cfg.qdrant_force_recreate is False

    def test_invalid_int_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
        with pytest.raises(ValueError):
            PipelineConfig.from_env()
