"""A client of the last release and a client of the next one work together.

Both clients run today's code. ``next_protocol`` makes the registry speak a
test-only next syft protocol, and ``pin_syft_protocol`` holds one client at the
released one. Each side then negotiates the released protocol, and a behaviour
that the next protocol adds stays off on both sides.
"""

from pathlib import Path

import pytest
from syft_rds import SyftRDSClient

from syft.migrations import client_registry
from syft.sync.syftbox_manager import SyftboxManager

RESULT_SCRIPT = "mkdir -p code/outputs && echo 42 > code/outputs/result.txt"


@pytest.mark.parametrize("held", ["do", "ds"])
def test_held_client_exchanges_job_at_released_protocol(
    held, next_protocol, pin_syft_protocol
):
    ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    pin_syft_protocol({"do": do, "ds": ds}[held].sync_engine, next_protocol.released)
    ds.load_peers()
    do.load_peers()
    ds_router = ds.sync_engine._connection_router
    do_router = do.sync_engine._connection_router

    released = next_protocol.released
    assert ds_router.negotiated_protocol_version_for_peer(do.email) == released
    assert do_router.negotiated_protocol_version_for_peer(ds.email) == released
    assert not ds_router.pair_speaks_protocol(do.email, next_protocol.next)
    assert not do_router.pair_speaks_protocol(ds.email, next_protocol.next)

    ds.submit_bash_job(do.email, RESULT_SCRIPT, "skew.job")
    do.sync()
    do.job_client.jobs[0].approve()
    do.process_approved_jobs(stream_output=False, share_outputs_with_submitter=True)
    do.sync()
    ds.sync()

    [output_path] = ds.job_client.jobs[-1].output_paths
    assert Path(output_path).read_text().strip() == "42"


def test_pair_on_next_protocol_speaks_it(next_protocol):
    # The control: without a held client both sides speak the next protocol, so
    # the test above measures the pin and not a broken switch.
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    ds_router = ds_manager._connection_router
    do_router = do_manager._connection_router

    assert ds_router.pair_speaks_protocol(do_manager.email, next_protocol.next)
    assert do_router.pair_speaks_protocol(ds_manager.email, next_protocol.next)
    # A pair on a protocol also speaks every earlier one.
    assert ds_router.pair_speaks_protocol(do_manager.email, next_protocol.released)
    assert do_router.pair_speaks_protocol(ds_manager.email, next_protocol.released)


def test_unknown_peer_speaks_no_protocol(advertised_schemas):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    advertised_schemas[ds_manager.email, "syft"] = None
    do_router = do_manager._connection_router

    assert do_router.negotiated_protocol_version_for_peer(ds_manager.email) is None
    assert not do_router.pair_speaks_protocol(
        ds_manager.email, client_registry.protocol_version
    )
