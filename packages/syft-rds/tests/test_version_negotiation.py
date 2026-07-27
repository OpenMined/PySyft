"""Version gating on the RDS job submission and execution paths."""

import pytest

from syft_client.sync.version.exceptions import VersionUnknownError
from syft_client.sync.version.version_info import VersionInfo
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
            stream_output=True, timeout=None, skip_job_names=None, **kwargs
        ):
            executed_jobs.append(skip_job_names)

        do.job_runner.process_approved_jobs = mock_process_approved_jobs

        do.process_approved_jobs(force_execution=True)

        assert len(executed_jobs) == 1
        assert executed_jobs[0] is None  # No jobs skipped when force=True
