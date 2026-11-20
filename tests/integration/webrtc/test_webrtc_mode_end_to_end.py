"""A DO and a DS in WebRTC mode: peering through a signaling server in this
process, and messages over real WebRTC on this machine. No Google Drive, mock
or real, takes part."""

import asyncio
import time

import pytest

from syft.sync.connections.config import ConnectionRouterConfig
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.syftbox_manager import SyftboxManager
from tests.integration.webrtc.utils import close, flush
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access
from tests.unit.webrtc.signaling_server_thread import (
    ACCESS_KEY,
    DELIVERY_SECONDS,
    sign_in_and_leave,
    signaling_server_in_thread,
)


@pytest.fixture
def server_url():
    with signaling_server_in_thread() as url:
        yield url


def build_webrtc_pair(
    url: str, add_peers: bool
) -> tuple[SyftboxManager, SyftboxManager]:
    """(ds_manager, do_manager), built only from a WebRTCConfig."""
    webrtc = WebRTCConfig(signaling_url=url, access_key=ACCESS_KEY)
    router = ConnectionRouterConfig.one_connection(webrtc)
    do_config, ds_config = SyftboxManager._mock_pair_configs(use_in_memory_cache=False)
    do_config = do_config.model_copy(update={"connection_router_config": router})
    ds_config = ds_config.model_copy(update={"connection_router_config": router})
    return SyftboxManager._test_pair(
        do_config,
        SyftboxManager._connections_from_config(do_config),
        ds_config,
        SyftboxManager._connections_from_config(ds_config),
        add_peers=add_peers,
    )


@pytest.fixture
def webrtc_pair(server_url):
    """(ds_manager, do_manager), peered."""
    ds_manager, do_manager = build_webrtc_pair(server_url, add_peers=True)
    yield ds_manager, do_manager
    close(ds_manager, do_manager)


def test_job_and_result_go_over_webrtc_with_no_drive(webrtc_pair):
    ds_manager, do_manager = webrtc_pair
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


def test_do_that_accepts_with_add_peer_gets_ds_job(server_url):
    ds_manager, do_manager = build_webrtc_pair(server_url, add_peers=False)
    try:
        ds_manager.add_peer(do_manager.email, sync=False)
        do_manager.load_peers()
        do_manager.add_peer(ds_manager.email, sync=False)
        ds_manager.load_peers()
        grant_job_inbox_access(do_manager, ds_manager.email)
        job_path = path_for_job(do_manager.email, ds_manager.email)

        ds_manager._send_file_change(job_path, "Hello, world!")
        flush(ds_manager)
        do_manager.sync()

        assert (do_manager.syftbox_folder / job_path).read_text() == "Hello, world!"
    finally:
        close(ds_manager, do_manager)


def test_sync_finishes_pass_before_it_reports_ended_session(webrtc_pair, server_url):
    ds_manager, do_manager = webrtc_pair
    grant_job_inbox_access(do_manager, ds_manager.email)
    job_path = path_for_job(do_manager.email, ds_manager.email)
    ds_manager._send_file_change(job_path, "Hello, world!")
    flush(ds_manager)
    asyncio.run(sign_in_and_leave(server_url, do_manager.email))
    time.sleep(DELIVERY_SECONDS)

    with pytest.raises(ConnectionError, match="signaling session ended"):
        do_manager.sync()

    assert (do_manager.syftbox_folder / job_path).read_text() == "Hello, world!"
