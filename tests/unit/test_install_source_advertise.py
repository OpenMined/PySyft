"""Tests for DO advertising syft-client install source via VersionInfo.

Covers:
- VersionInfo carries the new field and round-trips through JSON
- Backward compat: JSON missing the new field parses with field=None
- End-to-end: after pair setup, DS's JobClient knows DO's install source
- submit_python_job bakes the DO's source into run.sh
- Fallback path: DS warns and falls back to local detection when DO did not advertise
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from syft_client.sync.syftbox_manager import SyftboxManager
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


class TestEndToEndPeerInstallSourcePropagation:
    def test_ds_job_client_learns_do_install_source(self, monkeypatch):
        monkeypatch.setenv("SYFT_CLIENT_INSTALL_SOURCE", "/do/local/syft-client")

        ds, do = SyftboxManager.pair_with_mock_drive_service_connection()

        assert do.email in ds.job_client.peer_install_sources
        assert ds.job_client.peer_install_sources[do.email] == "/do/local/syft-client"

    def test_submit_python_job_bakes_do_source_into_run_sh(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYFT_CLIENT_INSTALL_SOURCE", "/do/local/syft-client")

        ds, do = SyftboxManager.pair_with_mock_drive_service_connection()

        # Minimal job: a single Python file
        code = tmp_path / "main.py"
        code.write_text("print('hello')\n")

        job_dir = ds.job_client.submit_python_job(
            user=do.email, code_path=str(code), job_name="my-test-job"
        )

        run_sh = (Path(job_dir) / "run.sh").read_text()
        # The DO's advertised source must appear in the uv pip install line,
        # and the DS's local source must NOT be substituted instead.
        assert "/do/local/syft-client" in run_sh
        assert "uv pip install" in run_sh


class TestFallbackWhenDoDidNotAdvertise:
    def test_falls_back_and_warns_when_peer_has_no_source(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYFT_CLIENT_INSTALL_SOURCE", "/ds/local/syft-client")

        ds, do = SyftboxManager.pair_with_mock_drive_service_connection()

        # Simulate an older DO: clear the advertised source from DS's JobClient.
        ds.job_client.peer_install_sources.pop(do.email, None)

        code = tmp_path / "main.py"
        code.write_text("print('hello')\n")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            job_dir = ds.job_client.submit_python_job(
                user=do.email,
                code_path=str(code),
                job_name="fallback-job",
            )

        # A prominent warning must have been emitted.
        messages = [str(w.message) for w in caught]
        assert any("No syft-client install source advertised" in m for m in messages), (
            f"Expected fallback warning, got: {messages}"
        )

        # The DS's local detection result should be used as the fallback.
        run_sh = (Path(job_dir) / "run.sh").read_text()
        assert "/ds/local/syft-client" in run_sh


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
