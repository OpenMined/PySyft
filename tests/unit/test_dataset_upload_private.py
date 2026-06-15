from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_datasets.dataset_manager import PRIVATE_DATASET_COLLECTION_PREFIX
from tests.unit.utils import create_tmp_dataset_files


class TestDatasetUploadPrivate:
    def test_ds_cannot_see_private_data(self):
        """When DO creates a dataset with upload_private=True,
        DS should see mock data but NOT private data."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            use_in_memory_cache=False,
        )

        mock_path, private_path, readme_path = create_tmp_dataset_files()

        do_manager.create_dataset(
            name="testdataset",
            mock_path=mock_path,
            private_path=private_path,
            summary="Test dataset",
            readme_path=readme_path,
            users=[ds_manager.email],
            upload_private=True,
        )

        # DS syncs
        ds_manager.sync()

        # DS should see the mock dataset
        ds_datasets = ds_manager.datasets.get_all()
        assert len(ds_datasets) == 1

        dataset = ds_manager.datasets.get("testdataset", datasite=do_manager.email)
        assert dataset is not None
        assert len(dataset.mock_files) > 0

        # DS should NOT see any private collections via the connection
        private_collections = (
            ds_manager._connection_router.owner_list_private_dataset_collections()
        )
        assert len(private_collections) == 0

        # Verify private collection folder exists on GDrive (visible to DO)
        do_private_collections = (
            do_manager._connection_router.owner_list_private_dataset_collections()
        )
        assert len(do_private_collections) == 1
        assert do_private_collections[0].tag == "testdataset"

    def test_do_cold_start_restores_private_data(self):
        """When DO creates a dataset with upload_private=True, then loses local data,
        a fresh sync should restore the private files."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            use_in_memory_cache=False,
        )

        mock_path, private_path, readme_path = create_tmp_dataset_files()

        do_manager.create_dataset(
            name="testdataset",
            mock_path=mock_path,
            private_path=private_path,
            summary="Test dataset",
            readme_path=readme_path,
            users=[ds_manager.email],
            upload_private=True,
        )

        # Verify private data exists locally
        dataset = do_manager.datasets.get("testdataset", datasite=do_manager.email)
        private_dir = dataset.private_dir
        assert private_dir.exists()
        private_files_before = {f.name for f in private_dir.iterdir()}
        assert len(private_files_before) > 0

        # Simulate cold start: delete local private dir and reset sync state
        import shutil

        shutil.rmtree(private_dir)
        assert not private_dir.exists()
        do_manager.datasite_owner_syncer.initial_sync_done = False

        # Sync again — should restore private data from GDrive
        do_manager.sync()

        # Private dir should be restored with all files
        assert private_dir.exists()
        restored_files = {f.name for f in private_dir.iterdir()}
        assert restored_files == private_files_before

    def test_default_behavior_no_private_upload(self):
        """When upload_private is not set, no private collection should be created."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            use_in_memory_cache=False,
        )

        mock_path, private_path, readme_path = create_tmp_dataset_files()

        do_manager.create_dataset(
            name="testdataset",
            mock_path=mock_path,
            private_path=private_path,
            summary="Test dataset",
            readme_path=readme_path,
            users=[ds_manager.email],
        )

        # No private collection should exist
        private_collections = (
            do_manager._connection_router.owner_list_private_dataset_collections()
        )
        assert len(private_collections) == 0

        # Mock collection should still exist
        mock_collections = (
            do_manager._connection_router.owner_list_dataset_collections()
        )
        assert "testdataset" in mock_collections

    def test_ds_cannot_find_private_folders_via_gdrive_query(self):
        """DS searching GDrive directly for private collection prefix should find nothing."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            use_in_memory_cache=False,
        )

        mock_path, private_path, readme_path = create_tmp_dataset_files()

        do_manager.create_dataset(
            name="testdataset",
            mock_path=mock_path,
            private_path=private_path,
            summary="Test dataset",
            readme_path=readme_path,
            users=[ds_manager.email],
            upload_private=True,
        )

        # Get the DS's raw GDrive connection and search for private folders
        ds_connection: GDriveConnection = ds_manager._connection_router.connections[0]
        results = (
            ds_connection.drive_service.files()
            .list(
                q=(
                    f"name contains '{PRIVATE_DATASET_COLLECTION_PREFIX}_' "
                    f"and trashed=false"
                ),
                fields="files(id,name)",
            )
            .execute()
        )
        assert len(results.get("files", [])) == 0

        # DO should find it with the same query
        do_connection: GDriveConnection = do_manager._connection_router.connections[0]
        do_results = (
            do_connection.drive_service.files()
            .list(
                q=(
                    f"name contains '{PRIVATE_DATASET_COLLECTION_PREFIX}_' "
                    f"and trashed=false"
                ),
                fields="files(id,name)",
            )
            .execute()
        )
        assert len(do_results.get("files", [])) == 1
