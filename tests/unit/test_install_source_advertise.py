"""VersionInfo carries the syft-client install source and round-trips through JSON."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from syft_client.sync.version.version_info import VersionInfo


@pytest.fixture(autouse=True)
def _clear_install_source_cache():
    """get_syft_client_install_source uses lru_cache; clear so env-var patches work."""
    from syft_job.install_source import get_syft_client_install_source

    get_syft_client_install_source.cache_clear()
    yield
    get_syft_client_install_source.cache_clear()


class TestVersionInfoCarriesInstallSource:
    def test_current_populates_install_source(self):
        v = VersionInfo.current()
        assert v.syft_client_install_source is not None
        assert isinstance(v.syft_client_install_source, str)
        assert len(v.syft_client_install_source) > 0

    def test_current_uses_env_var_override(self, monkeypatch):
        monkeypatch.setenv("SYFT_CLIENT_INSTALL_SOURCE", "/do/local/path")
        # cache cleared by fixture
        v = VersionInfo.current()
        assert v.syft_client_install_source == "/do/local/path"

    def test_json_roundtrip_preserves_install_source(self, monkeypatch):
        monkeypatch.setenv("SYFT_CLIENT_INSTALL_SOURCE", "/do/local/path")
        v = VersionInfo.current()
        restored = VersionInfo.from_json(v.to_json())
        assert restored.syft_client_install_source == "/do/local/path"

    def test_missing_field_parses_as_none_backward_compat(self):
        # Simulate JSON written by an older client (no install_source key)
        base = VersionInfo.current()
        payload = json.loads(base.to_json())
        payload.pop("syft_client_install_source", None)
        legacy_json = json.dumps(payload)

        restored = VersionInfo.from_json(legacy_json)
        assert restored.syft_client_install_source is None
        assert restored.syft_client_version == base.syft_client_version


class TestVersionInfoCurrentNeverRaises:
    def test_current_returns_versioninfo_even_if_detection_fails(self):
        # Force the install-source helper to blow up; current() must still return
        # a usable VersionInfo with install_source=None.
        with patch(
            "syft_job.install_source.get_syft_client_install_source",
            side_effect=RuntimeError("boom"),
        ):
            v = VersionInfo.current()
        assert isinstance(v, VersionInfo)
        assert v.syft_client_install_source is None
