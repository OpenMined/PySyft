"""Unit tests for version negotiation feature."""

import logging

import pytest
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.version.exceptions import (
    VersionMismatchError,
    VersionUnknownError,
)
from syft_client.sync.version.peer_manager import CompatAction
from syft_client.sync.version.version_info import CompatibilityStatus, VersionInfo


def build_client_version(client_version: str) -> VersionInfo:
    """Build a VersionInfo for a given client version, leaving other fields as-current."""
    base = VersionInfo.current()
    return VersionInfo(
        syft_client_version=client_version,
        min_supported_syft_client_version=base.min_supported_syft_client_version,
        protocol_version=base.protocol_version,
        min_supported_protocol_version=base.min_supported_protocol_version,
        updated_at=base.updated_at,
    )


class TestVersionInfo:
    """Tests for VersionInfo model."""

    def test_current_version_creates_valid_info(self):
        version_info = VersionInfo.current()
        assert version_info.syft_client_version is not None
        assert version_info.protocol_version is not None
        assert version_info.min_supported_syft_client_version is not None
        assert version_info.min_supported_protocol_version is not None
        assert version_info.updated_at is not None

    def test_version_info_json_roundtrip(self):
        original = VersionInfo.current()
        json_str = original.to_json()
        restored = VersionInfo.from_json(json_str)

        assert restored.syft_client_version == original.syft_client_version
        assert restored.protocol_version == original.protocol_version
        assert (
            restored.min_supported_syft_client_version
            == original.min_supported_syft_client_version
        )
        assert (
            restored.min_supported_protocol_version
            == original.min_supported_protocol_version
        )

    def test_compatible_versions_match(self):
        v1 = VersionInfo.current()
        v2 = VersionInfo.current()
        assert v1.is_compatible_with(v2) is True
        assert v2.is_compatible_with(v1) is True

    def test_minor_version_mismatch_is_incompatible(self):
        v1 = build_client_version("0.1.113")
        v2 = build_client_version("0.0.1")
        assert v1.is_compatible_with(v2) is False

    def test_protocol_only_diff_is_compatible_now(self):
        """Protocol_version differences alone no longer block compatibility."""
        base = VersionInfo.current()
        v_diff_protocol = VersionInfo(
            syft_client_version=base.syft_client_version,
            min_supported_syft_client_version=base.min_supported_syft_client_version,
            protocol_version="0.0.1",
            min_supported_protocol_version=base.min_supported_protocol_version,
            updated_at=base.updated_at,
        )
        assert base.is_compatible_with(v_diff_protocol) is True

    def test_get_incompatibility_reason_client(self):
        v1 = build_client_version("0.1.113")
        v2 = build_client_version("0.0.1")
        reason = v1.get_incompatibility_reason(v2)
        assert reason is not None
        assert "client" in reason.lower()


class TestCompatibilityStatus:
    """Tests for CompatibilityStatus enum and compatibility_status_with."""

    def test_same_version(self):
        assert build_client_version("0.1.113").compatibility_status_with(
            build_client_version("0.1.113")
        ) == (CompatibilityStatus.SAME)

    def test_patch_diff(self):
        assert build_client_version("0.1.113").compatibility_status_with(
            build_client_version("0.1.114")
        ) == (CompatibilityStatus.PATCH_DIFF)

    def test_minor_diff_is_incompatible(self):
        assert build_client_version("0.1.113").compatibility_status_with(
            build_client_version("0.2.0")
        ) == (CompatibilityStatus.INCOMPATIBLE)

    def test_major_diff_is_incompatible(self):
        assert build_client_version("0.1.113").compatibility_status_with(
            build_client_version("1.0.0")
        ) == (CompatibilityStatus.INCOMPATIBLE)

    def test_unknown_when_other_none(self):
        assert build_client_version("0.1.113").compatibility_status_with(None) == (
            CompatibilityStatus.UNKNOWN
        )

    def test_patch_warning_text_only_for_patch_diff(self):
        assert (
            build_client_version("0.1.113").get_patch_warning_text(
                build_client_version("0.1.113")
            )
            is None
        )
        assert (
            build_client_version("0.1.113").get_patch_warning_text(
                build_client_version("0.2.0")
            )
            is None
        )
        warning = build_client_version("0.1.113").get_patch_warning_text(
            build_client_version("0.1.114")
        )
        assert warning is not None
        assert "0.1.113" in warning and "0.1.114" in warning

    def test_is_compatible_with_includes_patch_diff(self):
        assert (
            build_client_version("0.1.113").is_compatible_with(
                build_client_version("0.1.114")
            )
            is True
        )
        assert (
            build_client_version("0.1.113").is_compatible_with(
                build_client_version("0.2.0")
            )
            is False
        )
        assert build_client_version("0.1.113").is_compatible_with(None) is False


class TestPeerManager:
    """Tests for PeerManager."""

    def test_peer_manager_writes_own_version(self):
        ds_manager, do_manager = (
            SyftboxManager.pair_with_mock_drive_service_connection()
        )

        ds_version = do_manager.peer_manager.load_peer_version(ds_manager.email)
        do_version = ds_manager.peer_manager.load_peer_version(do_manager.email)

        assert ds_version is not None
        assert do_version is not None

    def test_version_shared_on_add_peer(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            add_peers=False
        )

        ds_version_before = do_manager.peer_manager.load_peer_version(ds_manager.email)
        assert ds_version_before is None

        ds_manager.add_peer(do_manager.email)

        ds_version_after = do_manager.peer_manager.load_peer_version(ds_manager.email)
        assert ds_version_after is not None

    def test_version_shared_on_approve_peer(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            add_peers=False
        )

        ds_manager.add_peer(do_manager.email)
        do_manager.load_peers()

        do_version_before = ds_manager.peer_manager.load_peer_version(do_manager.email)
        assert do_version_before is None

        do_manager.approve_peer_request(ds_manager.email)

        do_version_after = ds_manager.peer_manager.load_peer_version(do_manager.email)
        assert do_version_after is not None

    def test_load_peer_version(self):
        ds_manager, do_manager = (
            SyftboxManager.pair_with_mock_drive_service_connection()
        )

        version = ds_manager.peer_manager.load_peer_version(do_manager.email)
        assert version is not None
        assert version.syft_client_version == VersionInfo.current().syft_client_version

    def test_load_peer_version_without_permission(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            add_peers=False
        )

        version = ds_manager.peer_manager.load_peer_version(do_manager.email)
        assert version is None

    def test_load_peer_versions_parallel(self):
        from syft_client.sync.connections.drive.gdrive_transport import (
            SYFT_VERSION_FILE,
        )
        from syft_client.sync.connections.drive.mock_drive_service import (
            MockDriveFile,
            MockPermission,
        )

        ds_manager, do_manager = (
            SyftboxManager.pair_with_mock_drive_service_connection()
        )

        backing_store = ds_manager._connection_router.connections[
            0
        ].drive_service._backing_store

        third_peer_email = "third_peer@test.com"
        third_peer_version = VersionInfo.current()
        version_file = MockDriveFile(
            name=SYFT_VERSION_FILE,
            mimeType="application/json",
            parents=[],
            owners=[{"emailAddress": third_peer_email}],
            content=third_peer_version.to_json(),
        )
        backing_store.add_file(version_file)

        backing_store.add_permission(
            version_file.id,
            MockPermission(
                type="user",
                role="reader",
                emailAddress=ds_manager.email,
            ),
        )

        peer_emails = [do_manager.email, third_peer_email]
        versions = ds_manager.peer_manager.load_peer_versions_parallel(
            peer_emails, force=True
        )

        assert len(versions) == 2
        assert versions[do_manager.email] is not None
        assert versions[third_peer_email] is not None
        assert (
            versions[third_peer_email].syft_client_version
            == third_peer_version.syft_client_version
        )


class TestForceSubmission:
    """Tests for force_submission parameter."""

    def test_job_submission_blocked_without_version(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            add_peers=False,
            sync_automatically=False,
            check_versions=True,
        )

        ds_manager.add_peer(do_manager.email)

        test_py_path = "/tmp/test_version.py"
        with open(test_py_path, "w") as f:
            f.write('print("hello")')

        with pytest.raises(VersionUnknownError):
            ds_manager.submit_python_job(
                user=do_manager.email,
                code_path=test_py_path,
                job_name="test.job",
            )

    def test_job_submission_allowed_with_force(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            add_peers=False,
            sync_automatically=False,
            check_versions=True,
        )

        ds_manager.add_peer(do_manager.email)

        test_py_path = "/tmp/test_version_force.py"
        with open(test_py_path, "w") as f:
            f.write('print("hello")')

        with pytest.raises(VersionUnknownError):
            ds_manager.submit_python_job(
                user=do_manager.email,
                code_path=test_py_path,
                job_name="test.fail.job",
            )

        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=test_py_path,
            job_name="test.force.job",
            force_submission=True,
        )

        job_dir = (
            ds_manager.syftbox_folder
            / do_manager.email
            / "app_data"
            / "job"
            / "inbox"
            / ds_manager.email
            / "test.force.job"
        )
        assert job_dir.exists()


def _set_peer_version(manager: SyftboxManager, peer_email: str, version: VersionInfo):
    """Override the cached version for a peer without a Drive round-trip."""
    peer = manager.peer_manager.get_cached_peer(peer_email)
    assert peer is not None, f"peer {peer_email} not in store"
    peer.version = version


def _patch_plus_one_version() -> VersionInfo:
    """Return a VersionInfo that bumps the current patch level by one."""
    parts = VersionInfo.current().syft_client_version.split(".")
    return build_client_version(f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}")


class TestPatchTolerance:
    """Patch-only differences: DSes log but don't skip; DOs skip by default."""

    def test_patch_diff_does_not_skip_in_sync_for_ds(self, caplog):
        """DS-side default: patch differences log but do not skip."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(ds_manager, do_manager.email, _patch_plus_one_version())

        with caplog.at_level(logging.INFO, logger="syft_client"):
            compatible = ds_manager.peer_manager.get_compatible_peer_emails_for_syncing(
                [do_manager.email]
            )
        assert do_manager.email in compatible
        assert any("patch" in r.getMessage().lower() for r in caplog.records)

    def test_patch_diff_skips_in_sync_for_do_default(self, caplog):
        """DO-side default: patch differences cause the peer to be skipped."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, _patch_plus_one_version())

        with caplog.at_level(logging.INFO, logger="syft_client"):
            compatible = do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
                [ds_manager.email]
            )
        assert ds_manager.email not in compatible
        assert any("patch" in r.getMessage().lower() for r in caplog.records)

    def test_patch_diff_allows_job_submission(self, caplog):
        """DS submitting to a patch-drifted DO does not raise (DS default = no skip)."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(ds_manager, do_manager.email, _patch_plus_one_version())

        with caplog.at_level(logging.INFO, logger="syft_client"):
            result = ds_manager.peer_manager.get_peer_compatibility_status(
                do_manager.email, action=CompatAction.SUBMIT
            )
            result.raise_on_skip(operation="submit job")
            result.maybe_warn()
        assert any("patch" in r.getMessage().lower() for r in caplog.records)

    def test_patch_diff_do_force_ignore_proceeds(self, caplog):
        """DO with force_ignore_peer_version proceeds despite patch drift."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, _patch_plus_one_version())
        do_manager.peer_manager.force_ignore_peer_version = True

        with caplog.at_level(logging.INFO, logger="syft_client"):
            compatible = do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
                [ds_manager.email]
            )
        assert ds_manager.email in compatible
        assert any(
            "proceeding anyway" in r.getMessage().lower() for r in caplog.records
        )

    def test_patch_diff_do_per_call_ignore_proceeds(self):
        """DO with per-call ignore_peer_version proceeds despite patch drift."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, _patch_plus_one_version())

        compatible = do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
            [ds_manager.email], ignore_peer_version=True
        )
        assert ds_manager.email in compatible

    def test_patch_diff_do_explicit_false_does_not_skip(self):
        """Setting skip_peer_on_patch_version_diff=False on a DO disables the skip."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, _patch_plus_one_version())
        do_manager.peer_manager.skip_peer_on_patch_version_diff = False

        compatible = do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
            [ds_manager.email]
        )
        assert ds_manager.email in compatible


class TestPeerManagerConfigPatchSkipDefault:
    """Tests for the role-derived default of skip_peer_on_patch_version_diff."""

    def _make_config(self, **kwargs):
        from pathlib import Path

        from syft_client.sync.version.peer_manager import PeerManagerConfig

        return PeerManagerConfig(syftbox_folder=Path("/tmp/syftbox"), **kwargs)

    def test_do_defaults_to_skip(self):
        cfg = self._make_config(has_do_role=True)
        assert cfg.skip_peer_on_patch_version_diff is True

    def test_ds_only_defaults_to_no_skip(self):
        cfg = self._make_config(has_ds_role=True)
        assert cfg.skip_peer_on_patch_version_diff is False

    def test_dual_role_defaults_to_skip(self):
        cfg = self._make_config(has_do_role=True, has_ds_role=True)
        assert cfg.skip_peer_on_patch_version_diff is True

    def test_explicit_false_on_do_is_preserved(self):
        cfg = self._make_config(has_do_role=True, skip_peer_on_patch_version_diff=False)
        assert cfg.skip_peer_on_patch_version_diff is False

    def test_explicit_true_on_ds_is_preserved(self):
        cfg = self._make_config(has_ds_role=True, skip_peer_on_patch_version_diff=True)
        assert cfg.skip_peer_on_patch_version_diff is True


class TestForceAllowIncompatiblePeers:
    """Tests for force_ignore_peer_version and per-call ignore_peer_version."""

    def test_incompatible_peer_skipped_by_default(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, build_client_version("99.0.0"))

        do_manager.peer_manager.suppress_version_warnings = True
        compatible = do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
            [ds_manager.email]
        )
        assert ds_manager.email not in compatible

    def test_force_allow_includes_incompatible_peer(self, caplog):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, build_client_version("99.0.0"))

        do_manager.peer_manager.force_ignore_peer_version = True
        with caplog.at_level(logging.INFO, logger="syft_client"):
            compatible = do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
                [ds_manager.email]
            )
        assert ds_manager.email in compatible
        assert any(
            "proceeding anyway" in r.getMessage().lower() for r in caplog.records
        )

    def test_per_call_ignore_peer_version_includes_peer(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, build_client_version("99.0.0"))

        compatible = do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
            [ds_manager.email], ignore_peer_version=True
        )
        assert ds_manager.email in compatible

    def test_per_call_ignore_peer_version_in_submit(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(ds_manager, do_manager.email, build_client_version("99.0.0"))

        with pytest.raises(VersionMismatchError):
            result = ds_manager.peer_manager.get_peer_compatibility_status(
                do_manager.email, action=CompatAction.SUBMIT
            )
            result.raise_on_skip(operation="submit job")

        # With per-call override, should not raise
        result = ds_manager.peer_manager.get_peer_compatibility_status(
            do_manager.email,
            action=CompatAction.SUBMIT,
            ignore_peer_version=True,
        )
        result.raise_on_skip(operation="submit job")

    def test_force_allow_in_submit(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(ds_manager, do_manager.email, build_client_version("99.0.0"))

        ds_manager.peer_manager.force_ignore_peer_version = True
        # Should not raise
        result = ds_manager.peer_manager.get_peer_compatibility_status(
            do_manager.email, action=CompatAction.SUBMIT
        )
        result.raise_on_skip(operation="submit job")


class TestVersionMismatchBehavior:
    """Tests for version mismatch behavior during operations."""

    def test_sync_skips_incompatible_peers(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )
        _set_peer_version(do_manager, ds_manager.email, build_client_version("0.0.1"))

        do_manager.peer_manager.suppress_version_warnings = True
        compatible_peers = (
            do_manager.peer_manager.get_compatible_peer_emails_for_syncing(
                [ds_manager.email]
            )
        )
        assert ds_manager.email not in compatible_peers

    def test_job_execution_forced_with_incompatible_version(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            sync_automatically=False,
            use_in_memory_cache=False,
        )

        test_py_path = "/tmp/test_exec_force.py"
        with open(test_py_path, "w") as f:
            f.write('print("hello")')

        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=test_py_path,
            job_name="test.exec.force.job",
        )

        do_manager.sync()
        assert len(do_manager.job_client.jobs) == 1
        job = do_manager.job_client.jobs[0]
        job.approve()

        _set_peer_version(do_manager, ds_manager.email, build_client_version("0.0.1"))

        executed_jobs = []

        def mock_process_approved_jobs(
            stream_output=True, timeout=None, skip_job_names=None, **kwargs
        ):
            executed_jobs.append(skip_job_names)

        do_manager.job_runner.process_approved_jobs = mock_process_approved_jobs

        do_manager.process_approved_jobs(force_execution=True)

        assert len(executed_jobs) == 1
        assert executed_jobs[0] is None  # No jobs skipped when force=True

    def test_version_upgrade_breaks_communication(self):
        """Major-bump upgrade should now make peers incompatible."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            check_versions=True,
        )

        ds_manager.peer_manager.load_peer_version(do_manager.email)
        do_manager.peer_manager.load_peer_version(ds_manager.email)

        assert ds_manager.peer_manager.is_peer_version_compatible(do_manager.email)
        assert do_manager.peer_manager.is_peer_version_compatible(ds_manager.email)

        _set_peer_version(do_manager, ds_manager.email, build_client_version("99.0.0"))

        assert not do_manager.peer_manager.is_peer_version_compatible(ds_manager.email)
