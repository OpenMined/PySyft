"""Version gating on the RDS job submission and execution paths."""

from unittest.mock import patch

import pytest

from syft.sync.version.exceptions import VersionUnknownError
from syft.sync.version.peer_manager import PeerCompatibilityResult, PeerManager
from syft.sync.version.version_info import CompatibilityStatus, VersionInfo
from syft_rds import SyftRDSClient


def build_client_version(version: str) -> VersionInfo:
    return VersionInfo(
        syft_client_version=version,
        min_supported_syft_client_version=version,
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
    )


def _set_peer_version(client, peer_email: str, version: VersionInfo):
    """Override the cached version for a peer without a Drive round-trip."""
    peer = client.peer_manager.get_cached_peer(peer_email)
    assert peer is not None, f"peer {peer_email} not in store"
    peer.version = version


class TestForceSubmission:
    """Tests for force_submission parameter."""

    def test_job_submission_blocked_without_version(self):
        ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
            add_peers=False,
            sync_automatically=False,
            check_versions=True,
        )

        ds.add_peer(do.email)

        test_py_path = "/tmp/test_version.py"
        with open(test_py_path, "w") as f:
            f.write('print("hello")')

        with pytest.raises(VersionUnknownError):
            ds.submit_python_job(
                user=do.email,
                code_path=test_py_path,
                job_name="test.job",
            )

    def test_job_submission_allowed_with_force(self):
        ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
            add_peers=False,
            sync_automatically=False,
            check_versions=True,
        )

        ds.add_peer(do.email)

        test_py_path = "/tmp/test_version_force.py"
        with open(test_py_path, "w") as f:
            f.write('print("hello")')

        with pytest.raises(VersionUnknownError):
            ds.submit_python_job(
                user=do.email,
                code_path=test_py_path,
                job_name="test.fail.job",
            )

        ds.submit_python_job(
            user=do.email,
            code_path=test_py_path,
            job_name="test.force.job",
            force_submission=True,
        )

        job_dir = (
            ds.syftbox_folder
            / do.email
            / "app_data"
            / "job"
            / "inbox"
            / ds.email
            / "v1"
            / "test.force.job"
        )
        assert job_dir.exists()


class TestVersionMismatchBehavior:
    def test_job_execution_forced_with_incompatible_version(self):
        ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
            sync_automatically=False,
            use_in_memory_cache=False,
        )

        test_py_path = "/tmp/test_exec_force.py"
        with open(test_py_path, "w") as f:
            f.write('print("hello")')

        ds.submit_python_job(
            user=do.email,
            code_path=test_py_path,
            job_name="test.exec.force.job",
        )

        do.sync()
        assert len(do.job_client.jobs) == 1
        job = do.job_client.jobs[0]
        job.approve()

        _set_peer_version(do, ds.email, build_client_version("0.0.1"))

        executed_jobs = []

        def mock_process_approved_jobs(
            stream_output=True, timeout=None, skip_jobs=None, **kwargs
        ):
            executed_jobs.append(skip_jobs)

        do.job_runner.process_approved_jobs = mock_process_approved_jobs

        do.process_approved_jobs(force_execution=True)

        assert len(executed_jobs) == 1
        assert executed_jobs[0] is None  # No jobs skipped when force=True


class TestSkippedJobsAreReported:
    """process_approved_jobs must say which approved jobs it did not run.

    maybe_warn() logs the peer, not the job, so on its own it leaves the data
    owner with a job stuck at 'approved' and nothing naming it.
    """

    def test_skipped_job_name_and_reason_reach_stdout(self, tmp_path, capfd):
        ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
            use_in_memory_cache=False,
            sync_automatically=False,
        )

        code_path = tmp_path / "skipped.py"
        code_path.write_text('print("hello")')
        ds.submit_python_job(
            user=do.email, code_path=str(code_path), job_name="skipped.job"
        )
        do.sync()
        do.jobs["skipped.job"].approve()

        skip = PeerCompatibilityResult(
            peer_email=ds.email,
            status=CompatibilityStatus.INCOMPATIBLE,
            should_skip=True,
            explanation_skip=f"Skipping peer {ds.email}: incompatible version.",
        )
        capfd.readouterr()
        with patch.object(
            PeerManager, "get_peer_compatibility_status", return_value=skip
        ):
            do.process_approved_jobs()
        out, _ = capfd.readouterr()

        assert "skipped.job" in out
        assert ds.email in out
        assert "incompatible version" in out
        assert "ignore_peer_version=True" in out
        assert do.jobs["skipped.job"].status == "approved"
