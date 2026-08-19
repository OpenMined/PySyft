"""An outgoing sync message is downgraded to the peer's negotiated protocol.

The receive side already upgrades: every router receive path decodes through
load_as_latest, so an old blob reads on a new client. The send side is the
other half of that contract. A sender at a newer message version must write the
version the recipient's protocol supports, or the recipient cannot decode the
blob at all -- there is no newer class in its registry to load.

These tests drive the two send paths of the ConnectionRouter (DS -> DO
proposals, DO -> DS events) against a peer that advertises an older syft-client
protocol, and read the raw bytes off the mock drive to see which version was
actually put on the wire.
"""

import json
import logging

import pytest
from syft_client.migrations import client_registry
from syft_client.sync.events.file_change_event import FileChangeEventsMessageV1
from syft_client.sync.messages.proposed_filechange import ProposedFileChangesMessageV1
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.utils.syftbox_utils import uncompress_data
from syft_migration import MigrationError, ProtocolSchema

from tests.unit.utils import get_mock_events_messages, mock_message


def _client_schema(
    protocol_version: str, min_supported_version: str = "0"
) -> ProtocolSchema:
    # The slim form a peer advertises in its VersionInfo.
    return ProtocolSchema(
        protocol_name="syft-client",
        version=protocol_version,
        min_supported_version=min_supported_version,
        supported_versions={
            "VersionInfo": ["1"],
            "ProposedFileChangesMessage": ["1"],
            "FileChangeEventsMessage": ["1"],
        },
    )


@pytest.fixture
def pair():
    return SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )


@pytest.fixture
def v2_wire_envelopes():
    """Throwaway V2 envelope classes, as the next release would ship them.

    Subclassing a registered class inherits its registry, so these register
    into the global client_registry; the teardown pops them back out so no
    other test sees a version "2" (the registry has no deregister API).
    """

    class ProposedFileChangesMessageV2(ProposedFileChangesMessageV1):
        version: str = "2"

    class FileChangeEventsMessageV2(FileChangeEventsMessageV1):
        version: str = "2"

    for canonical_name, v1, v2 in (
        (
            "ProposedFileChangesMessage",
            ProposedFileChangesMessageV1,
            ProposedFileChangesMessageV2,
        ),
        (
            "FileChangeEventsMessage",
            FileChangeEventsMessageV1,
            FileChangeEventsMessageV2,
        ),
    ):
        client_registry.register_migration(
            canonical_name=canonical_name,
            from_version="1",
            to_version="2",
            fn=lambda obj, v2=v2: v2(**obj.model_dump(exclude={"version"})),
        )
        client_registry.register_migration(
            canonical_name=canonical_name,
            from_version="2",
            to_version="1",
            fn=lambda obj, v1=v1: v1(**obj.model_dump(exclude={"version"})),
        )

    yield ProposedFileChangesMessageV2, FileChangeEventsMessageV2

    for canonical_name in ("ProposedFileChangesMessage", "FileChangeEventsMessage"):
        client_registry.objects[canonical_name].pop("2", None)
        client_registry.migrations.get(canonical_name, {}).pop(("1", "2"), None)
        client_registry.migrations.get(canonical_name, {}).pop(("2", "1"), None)


def _raw_proposal_version(do_manager, ds_email: str) -> str:
    """The version field of the next proposal blob in the DO's inbox, unparsed."""
    raw, _ = do_manager._connection_router.connections[
        0
    ].owner_download_next_raw_proposed_message_from_inbox(ds_email)
    return json.loads(uncompress_data(raw))["version"]


def _raw_outbox_versions(ds_manager, do_email: str) -> list[str]:
    """The version fields of the DO's outbox blobs for us, unparsed."""
    raw_list = ds_manager._connection_router.connections[
        0
    ].watcher_download_raw_events_from_outbox(do_email, None)
    return [json.loads(uncompress_data(raw))["version"] for raw in raw_list]


def test_a_v2_proposal_for_a_protocol0_peer_downgrades_on_the_wire(
    pair, v2_wire_envelopes
):
    ds_manager, do_manager = pair
    v2_proposed, _ = v2_wire_envelopes
    # The DO advertises client protocol 0, as a client of 0.1.117 or earlier does.
    ds_manager.peer_manager.live_peer_schemas("syft-client")[do_manager.email] = (
        _client_schema("0")
    )

    message = v2_proposed(**mock_message().model_dump(exclude={"version"}))
    ds_manager._connection_router.watcher_send_proposed_file_changes_message(
        do_manager.email, message
    )

    assert _raw_proposal_version(do_manager, ds_manager.email) == "1", (
        "the peer's protocol supports message version 1 only, so the sender "
        "must downgrade before the blob goes up"
    )


def test_a_v2_events_message_for_a_protocol0_peer_downgrades_on_the_wire(
    pair, v2_wire_envelopes
):
    ds_manager, do_manager = pair
    _, v2_events = v2_wire_envelopes
    do_manager.peer_manager.live_peer_schemas("syft-client")[ds_manager.email] = (
        _client_schema("0")
    )

    message = v2_events(
        **get_mock_events_messages(1)[0].model_dump(exclude={"version"})
    )
    do_manager._connection_router.owner_write_event_messages_to_outbox(
        ds_manager.email, message
    )

    assert _raw_outbox_versions(ds_manager, do_manager.email) == ["1"]


def test_a_send_beyond_the_peers_floor_raises(pair):
    # A future peer that dropped support for our protocol. Sending anyway would
    # put up a blob the peer refuses; the negotiation must fail loudly instead.
    ds_manager, do_manager = pair
    ds_manager.peer_manager.live_peer_schemas("syft-client")[do_manager.email] = (
        _client_schema("2", min_supported_version="2")
    )

    with pytest.raises(MigrationError):
        ds_manager._connection_router.watcher_send_proposed_file_changes_message(
            do_manager.email, mock_message()
        )


def test_a_send_to_an_unknown_peer_warns_and_keeps_the_current_version(pair, caplog):
    # Same policy as jobs: a peer without a known schema is assumed to run the
    # current protocol, and the assumption is logged.
    ds_manager, do_manager = pair
    ds_manager.peer_manager.live_peer_schemas("syft-client").pop(do_manager.email, None)

    with caplog.at_level(logging.WARNING):
        ds_manager._connection_router.watcher_send_proposed_file_changes_message(
            do_manager.email, mock_message()
        )

    assert "No syft-client protocol schema known" in caplog.text
    assert _raw_proposal_version(do_manager, ds_manager.email) == "1"


def test_a_current_protocol_peer_gets_the_current_version(pair):
    # The control: a peer on our own protocol gets the current version, so the
    # tests above measure the downgrade and not a broken default.
    ds_manager, do_manager = pair
    ds_manager.peer_manager.live_peer_schemas("syft-client")[do_manager.email] = (
        _client_schema("1")
    )

    ds_manager._connection_router.watcher_send_proposed_file_changes_message(
        do_manager.email, mock_message()
    )

    assert _raw_proposal_version(do_manager, ds_manager.email) == "1"
