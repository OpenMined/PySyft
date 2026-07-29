"""DS JobClient picks up the DO's advertised syft-client install source."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from syft_rds import SyftRDSClient


@pytest.fixture(autouse=True)
def _clear_install_source_cache():
    """get_syft_client_install_source uses lru_cache; clear so env-var patches work."""
    from syft_job.install_source import get_syft_client_install_source

    get_syft_client_install_source.cache_clear()
    yield
    get_syft_client_install_source.cache_clear()


class TestEndToEndPeerInstallSourcePropagation:
    def test_ds_job_client_learns_do_install_source(self, monkeypatch):
        monkeypatch.setenv("SYFT_CLIENT_INSTALL_SOURCE", "/do/local/syft-client")

        ds, do = SyftRDSClient.pair_with_mock_drive_service_connection()

        assert do.email in ds.job_client.peer_install_sources
        assert ds.job_client.peer_install_sources[do.email] == "/do/local/syft-client"

    def test_submit_python_job_bakes_do_source_into_run_sh(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYFT_CLIENT_INSTALL_SOURCE", "/do/local/syft-client")

        ds, do = SyftRDSClient.pair_with_mock_drive_service_connection()

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

        ds, do = SyftRDSClient.pair_with_mock_drive_service_connection()

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
