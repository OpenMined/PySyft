"""Tests for create_dataset automatic cleanup on failure."""

import logging
from unittest.mock import patch

import pytest

from syft_client.sync.syftbox_manager import SyftboxManager
from tests.unit.utils import create_tmp_dataset_files


class TestCreateDatasetCleanup:
    """Tests that create_dataset cleans up partial state on failure."""

    def _make_do_manager(self):
        _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            use_in_memory_cache=False,
        )
        return do_manager

    def _dataset_kwargs(self, users=None):
        mock_path, private_path, readme_path = create_tmp_dataset_files()
        return dict(
            name="testdataset",
            mock_path=mock_path,
            private_path=private_path,
            summary="Test dataset",
            readme_path=readme_path,
            users=users or [],
        )

    def test_no_cleanup_when_local_create_fails(self):
        """If dataset_manager.create raises, nothing was created so nothing to clean."""
        do_manager = self._make_do_manager()

        with (
            patch.object(
                do_manager.dataset_manager,
                "create",
                side_effect=ValueError("bad input"),
            ),
            patch.object(
                do_manager, "_cleanup_failed_dataset_creation"
            ) as mock_cleanup,
        ):
            with pytest.raises(ValueError, match="bad input"):
                do_manager.create_dataset(**self._dataset_kwargs())

        # Cleanup called with nothing to clean
        mock_cleanup.assert_called_once_with(None, False, None, None)

    def test_cleanup_on_mock_upload_failure(self):
        """If mock upload fails, local dataset is cleaned up."""
        do_manager = self._make_do_manager()

        with patch.object(
            do_manager,
            "_upload_dataset_to_collection",
            side_effect=RuntimeError("upload failed"),
        ):
            with pytest.raises(RuntimeError, match="upload failed"):
                do_manager.create_dataset(**self._dataset_kwargs())

        # Local dataset should have been cleaned up
        datasets = do_manager.dataset_manager.get_all()
        assert len(datasets) == 0

    def test_cleanup_on_private_upload_failure(self):
        """If private upload fails, both GDrive mock folder and local dataset are cleaned up."""
        do_manager = self._make_do_manager()

        # Compute expected local paths before the test so we can verify deletion
        syftbox_config = do_manager.dataset_manager.syftbox_config
        mock_dir = syftbox_config.get_my_mock_dataset_dir("testdataset")
        private_metadata_dir = syftbox_config.private_dir_for_my_dataset("testdataset")

        with patch.object(
            do_manager,
            "_upload_private_dataset_to_collection",
            side_effect=RuntimeError("private upload failed"),
        ):
            with pytest.raises(RuntimeError, match="private upload failed"):
                do_manager.create_dataset(**self._dataset_kwargs(), upload_private=True)

        # Local dataset should have been cleaned up
        datasets = do_manager.dataset_manager.get_all()
        assert len(datasets) == 0

        # Local filesystem directories should not exist
        assert not mock_dir.exists(), f"mock_dir was not cleaned up: {mock_dir}"
        assert not private_metadata_dir.exists(), (
            f"private_metadata_dir was not cleaned up: {private_metadata_dir}"
        )

        # Mock GDrive folder should have been cleaned up too
        collections = do_manager._connection_router.owner_list_dataset_collections()
        assert "testdataset" not in collections

    def test_cleanup_on_sync_failure(self):
        """If sync fails, both GDrive folders and local dataset are cleaned up."""
        do_manager = self._make_do_manager()

        with patch.object(
            SyftboxManager, "sync", side_effect=RuntimeError("sync failed")
        ):
            with pytest.raises(RuntimeError, match="sync failed"):
                do_manager.create_dataset(**self._dataset_kwargs(), upload_private=True)

        # Everything should be cleaned up
        datasets = do_manager.dataset_manager.get_all()
        assert len(datasets) == 0

    def test_original_exception_propagates_when_cleanup_also_fails(self):
        """If cleanup itself fails, the original exception still propagates."""
        do_manager = self._make_do_manager()

        with (
            patch.object(
                do_manager,
                "_upload_dataset_to_collection",
                side_effect=RuntimeError("original error"),
            ),
            patch.object(
                do_manager.dataset_manager,
                "delete",
                side_effect=OSError("cleanup also broken"),
            ),
        ):
            with pytest.raises(RuntimeError, match="original error"):
                do_manager.create_dataset(**self._dataset_kwargs())

    def test_cleanup_logs_warning_on_failure(self, caplog):
        """Cleanup failures are logged as warnings, not raised."""
        do_manager = self._make_do_manager()

        with (
            patch.object(
                do_manager,
                "_upload_dataset_to_collection",
                side_effect=RuntimeError("upload failed"),
            ),
            patch.object(
                do_manager.dataset_manager,
                "delete",
                side_effect=OSError("delete broken"),
            ),
        ):
            with caplog.at_level(
                logging.WARNING, logger="syft_client.sync.syftbox_manager"
            ):
                with pytest.raises(RuntimeError, match="upload failed"):
                    do_manager.create_dataset(**self._dataset_kwargs())

            assert "Cleanup: failed to delete local dataset" in caplog.text

    def test_happy_path_no_cleanup(self):
        """Successful create_dataset doesn't trigger cleanup."""
        do_manager = self._make_do_manager()

        with patch.object(
            do_manager, "_cleanup_failed_dataset_creation"
        ) as mock_cleanup:
            dataset = do_manager.create_dataset(**self._dataset_kwargs())

        mock_cleanup.assert_not_called()
        assert dataset.name == "testdataset"
