"""Managers whose messages go over real WebRTC on this machine."""

from pathlib import Path

import pytest
from syft_rds import SyftRDSClient

from syft.sync.syftbox_manager import SyftboxManager
from tests.integration.webrtc.utils import close, flush
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access
from tests.unit.webrtc.test_webrtc_end_to_end import drive_message_files


@pytest.fixture
def pair():
    ds_manager, do_manager = (
        SyftboxManager.pair_with_mock_drive_and_local_webrtc_connection(
            use_in_memory_cache=False
        )
    )
    yield ds_manager, do_manager
    close(ds_manager, do_manager)


@pytest.fixture
def rds_pair():
    ds, do = SyftRDSClient.pair_with_mock_drive_and_local_webrtc_connection(
        use_in_memory_cache=False
    )
    yield ds, do
    close(ds.sync_engine, do.sync_engine)


def test_file_round_trip_goes_over_webrtc(pair):
    ds_manager, do_manager = pair
    grant_job_inbox_access(do_manager, ds_manager.email)
    job_path = path_for_job(do_manager.email, ds_manager.email)

    ds_manager._send_file_change(job_path, "Hello, world!")
    flush(ds_manager)
    do_manager.sync()

    assert (do_manager.syftbox_folder / job_path).read_text() == "Hello, world!"

    datasite_do = do_manager.syftbox_folder / do_manager.email
    do_manager.datasite_owner_syncer.perm_context.open("results/").grant_read_access(
        ds_manager.email
    )
    (datasite_do / "results" / "result.txt").write_text("I am a result")
    do_manager.sync()
    flush(do_manager)
    ds_manager.sync()

    result_on_ds = ds_manager.syftbox_folder / do_manager.email / "results/result.txt"
    assert result_on_ds.read_text() == "I am a result"
    assert drive_message_files(do_manager, ds_manager.email) == []


def test_job_round_trip_goes_over_webrtc(rds_pair):
    ds, do = rds_pair
    script = "mkdir -p code/outputs && echo 42 > code/outputs/result.txt"

    ds.submit_bash_job(do.email, script, "webrtc.job")
    flush(ds.sync_engine)
    do.sync()
    do.job_client.jobs[0].approve()
    do.process_approved_jobs(stream_output=False, share_outputs_with_submitter=True)
    do.sync()
    flush(do.sync_engine)
    ds.sync()

    [output_path] = ds.job_client.jobs[-1].output_paths
    assert Path(output_path).read_text().strip() == "42"
    assert drive_message_files(do.sync_engine, ds.email) == []
