"""Unit tests for app.config module."""

from __future__ import annotations

import pytest


class TestToBool:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "y", "on"])
    def test_truthy_values(self, value, to_bool_fn):
        assert to_bool_fn(value, False) is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "n", "off"])
    def test_falsy_values(self, value, to_bool_fn):
        assert to_bool_fn(value, True) is False

    def test_none_returns_default(self, to_bool_fn):
        assert to_bool_fn(None, True) is True
        assert to_bool_fn(None, False) is False

    def test_unknown_returns_default(self, to_bool_fn):
        assert to_bool_fn("maybe", True) is True
        assert to_bool_fn("maybe", False) is False

    def test_whitespace_stripped(self, to_bool_fn):
        assert to_bool_fn("  true  ", False) is True


class TestBackendConfig:
    def test_defaults(self, config_cls):
        config = config_cls()
        assert config.generate_model_name == "gemini"
        assert config.retrieval_top_k == 20
        assert config.rerank_top_k == 5
        assert config.enable_reranker is True
        assert config.agent_max_iterations == 5
        assert config.memory_max_recent_turns == 3
        assert config.memory_session_ttl == 1800
        assert config.official_site_allowlist == "fit.hcmus.edu.vn"

    def test_custom_values(self, config_cls):
        config = config_cls(
            agent_max_iterations=10,
            memory_max_recent_turns=5,
            memory_session_ttl=600,
        )
        assert config.agent_max_iterations == 10
        assert config.memory_max_recent_turns == 5
        assert config.memory_session_ttl == 600
