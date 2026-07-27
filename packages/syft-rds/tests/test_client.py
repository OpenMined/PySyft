"""Tests for the self-contained SyftRDSClient product."""

from syft_rds import SyftRDSClient


def test_rds_layer_supplies_collection_specs():
    """The composed DS sync engine's watcher cache received the dataset spec
    from the RDS layer (the RDS -> generic-engine spec-injection seam)."""
    from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX

    ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    watcher_cache = ds.sync_engine.datasite_watcher_syncer.datasite_watcher_cache
    specs = watcher_cache.collection_specs

    assert len(specs) > 0
    assert any(spec.prefix == DATASET_COLLECTION_PREFIX for spec in specs)


def test_dataset_creation_and_sync():
    """Datasets created by the DO are visible to the DS via mock drive."""
    from dataset_test_utils import create_tmp_dataset_files

    ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )

    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do.create_dataset(
        name="mock drive dataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test dataset via mock drive",
        readme_path=readme_path,
        tags=["test"],
        users=[ds.email],
    )

    assert len(do.datasets.get_all()) == 1

    ds.sync()

    assert len(ds.datasets.get_all()) == 1
    dataset = ds.datasets.get("mock drive dataset", datasite=do.email)
    assert dataset is not None
    assert len(dataset.mock_files) > 0


def test_delete_unversioned_state_removes_dataset_collections():
    """delete_unversioned_state clears both dataset collection folders."""
    from syft_datasets.dataset_manager import (
        DATASET_COLLECTION_PREFIX,
        PRIVATE_DATASET_COLLECTION_PREFIX,
    )
    from dataset_test_utils import create_tmp_dataset_files

    def query(conn, name_contains):
        results = (
            conn.drive_service.files()
            .list(
                q=f"name contains '{name_contains}' and trashed=false",
                fields="files(id, name)",
            )
            .execute()
        )
        return results.get("files", [])

    ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
        encryption=True,
    )

    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do.create_dataset(
        name="my dataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test",
        readme_path=readme_path,
        users=[ds.email],
        upload_private=True,
    )
    do.sync()

    conn = do.peer_manager.connection_router.connections[0]
    assert len(query(conn, DATASET_COLLECTION_PREFIX)) > 0
    assert len(query(conn, PRIVATE_DATASET_COLLECTION_PREFIX)) > 0

    conn.delete_unversioned_state()

    assert len(query(conn, DATASET_COLLECTION_PREFIX)) == 0
    assert len(query(conn, PRIVATE_DATASET_COLLECTION_PREFIX)) == 0


def test_dir_returns_only_public_api():
    _ds, do = SyftRDSClient.pair_with_mock_drive_service_connection()

    public_names = dir(do)

    # Identity + sync surface
    assert "email" in public_names
    assert "sync" in public_names
    assert "peers" in public_names
    assert "add_peer" in public_names

    # Composed engine and managers
    assert "sync_engine" in public_names
    assert "job_client" in public_names
    assert "dataset_manager" in public_names

    # Dataset/job surface
    assert "jobs" in public_names
    assert "datasets" in public_names
    assert "create_dataset" in public_names
    assert "submit_python_job" in public_names
    assert "submit_bash_job" in public_names
    assert "process_approved_jobs" in public_names

    # Hides Pydantic internals
    assert "model_dump" not in public_names
    assert "model_fields" not in public_names
    assert "model_validate" not in public_names

    # Hides internal helpers
    assert "model_post_init" not in public_names
    assert "from_config" not in public_names

    # Hidden attributes are still accessible
    assert do.email is not None
    assert callable(do.model_dump)


def test_encrypted_dataset_collection_syncs():
    """Dataset-collection sync-down works under encryption."""
    from dataset_test_utils import create_tmp_dataset_files
    from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX

    ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
        encryption=True,
    )

    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do.create_dataset(
        name="demo",
        mock_path=mock_path,
        private_path=private_path,
        summary="demo dataset",
        readme_path=readme_path,
        users=[ds.email],
        upload_private=True,
        sync=False,
    )
    do.sync()

    ds.sync()

    cr = ds.peer_manager.connection_router
    collections = cr.watcher_list_collections(DATASET_COLLECTION_PREFIX)
    do_collections = [c for c in collections if c["owner_email"] == do.email]
    assert do_collections, "DS does not see the DO's dataset collection"

    c = do_collections[0]
    files = cr.watcher_download_collection(
        DATASET_COLLECTION_PREFIX, c["tag"], c["content_hash"], do.email
    )
    assert files, "DS could not download the dataset collection files"
