from syft_client.sync.syftbox_manager import SyftboxManager


def test_dir_returns_only_public_api():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()

    public_names = dir(ds_manager)

    # Contains public API
    assert "email" in public_names
    assert "sync" in public_names
    assert "peers" in public_names
    assert "jobs" in public_names
    assert "datasets" in public_names
    assert "add_peer" in public_names
    assert "create_dataset" in public_names
    assert "submit_python_job" in public_names

    # Hides Pydantic internals
    assert "model_dump" not in public_names
    assert "model_fields" not in public_names
    assert "model_validate" not in public_names

    # Hides internal fields
    assert "datasite_owner_syncer" not in public_names
    assert "datasite_watcher_syncer" not in public_names
    assert "file_writer" not in public_names
    assert "job_file_change_handler" not in public_names

    # Hidden attributes are still accessible
    assert ds_manager.email is not None
    assert ds_manager.file_writer is not None
    assert callable(ds_manager.model_dump)
