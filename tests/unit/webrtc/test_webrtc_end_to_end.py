"""Managers that send their messages over WebRTC joined by FakePeerConnections."""

from pathlib import Path

from syft_rds import SyftRDSClient

from syft.sync.connections.drive.drive_connections import drive_cached_client_of
from syft.sync.syftbox_manager import SyftboxManager
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access


def drive_message_files(do_manager: SyftboxManager, ds_email: str) -> list[dict]:
    """The files in the Drive inbox, its archive and the outbox between the two
    peers. Over Drive a proposal lands in the inbox and is then archived."""
    client = drive_cached_client_of(do_manager._connection_router.connections)
    folder_ids = [
        client.get_own_datasite_inbox_id(ds_email),
        client.get_archive_folder_id_from_drive(ds_email),
        client.get_own_datasite_outbox_id(ds_email),
    ]
    return [
        f
        for folder_id in folder_ids
        if folder_id is not None
        for f in client.get_file_metadatas_from_folder(folder_id)
    ]


def test_file_round_trip_goes_over_webrtc():
    ds_manager, do_manager = (
        SyftboxManager.pair_with_mock_drive_and_mock_webrtc_connection(
            use_in_memory_cache=False
        )
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    job_path = path_for_job(do_manager.email, ds_manager.email)

    ds_manager._send_file_change(job_path, "Hello, world!")
    do_manager.sync()

    assert (do_manager.syftbox_folder / job_path).read_text() == "Hello, world!"

    datasite_do = do_manager.syftbox_folder / do_manager.email
    do_manager.datasite_owner_syncer.perm_context.open("results/").grant_read_access(
        ds_manager.email
    )
    (datasite_do / "results" / "result.txt").write_text("I am a result")
    do_manager.sync()
    ds_manager.sync()

    result_on_ds = ds_manager.syftbox_folder / do_manager.email / "results/result.txt"
    assert result_on_ds.read_text() == "I am a result"
    assert drive_message_files(do_manager, ds_manager.email) == []


def test_job_round_trip_goes_over_webrtc():
    ds, do = SyftRDSClient.pair_with_mock_drive_and_mock_webrtc_connection(
        use_in_memory_cache=False
    )
    script = "mkdir -p code/outputs && echo 42 > code/outputs/result.txt"

    ds.submit_bash_job(do.email, script, "webrtc.job")
    do.sync()
    do.job_client.jobs[0].approve()
    do.process_approved_jobs(stream_output=False, share_outputs_with_submitter=True)
    do.sync()
    ds.sync()

    [output_path] = ds.job_client.jobs[-1].output_paths
    assert Path(output_path).read_text().strip() == "42"
    assert drive_message_files(do.sync_engine, ds.email) == []
