"""An outgoing sync message is downgraded to the peer's negotiated protocol.

The receive side already upgrades: every router receive path decodes through
load_current, so an old blob reads on a new client. The send side is the
other half of that contract. A sender at a newer message version must write the
version the recipient's protocol supports, or the recipient cannot decode the
blob at all -- there is no newer class in its registry to load.

These tests drive the two send paths of the ConnectionRouter (DS -> DO
proposals, DO -> DS events) against a peer that advertises an older syft
protocol, and read the raw bytes off the mock drive to see which version was
actually put on the wire.

Every test runs with encryption off and on. The downgrade runs before the
encrypt, so the version on the wire sits inside the envelope, and a reader has
to open the envelope to see it.
"""

import json
import logging

import pytest
from syft.migrations import client_registry
from syft.sync.events.file_change_event import FileChangeEventsMessageV0
from syft.sync.messages.proposed_filechange import ProposedFileChangesMessageV0
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.connections.types import MessageKind
from syft.sync.utils.syftbox_utils import uncompress_data
from syft_migration import MigrationError, ProtocolSchema

from tests.unit.utils import get_mock_events_messages, mock_message


def _client_schema(
    protocol_version: str, min_supported_version: str = "0"
) -> ProtocolSchema:
    # The slim form a peer advertises in its VersionInfo.
    return ProtocolSchema(
        protocol_name="syft",
        version=protocol_version,
        min_supported_version=min_supported_version,
        supported_versions={
            "VersionInfo": ["0"],
            "ProposedFileChangesMessage": ["0"],
            "FileChangeEventsMessage": ["0"],
        },
    )


@pytest.fixture(params=[False, True], ids=["plaintext", "encrypted"])
def pair(request):
    return SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
        encryption=request.param,
    )


@pytest.fixture
def v1_wire_envelopes():
    """Throwaway V1 envelope classes, as the next release would ship them.

    Subclassing a registered class inherits its registry, so these register
    into the global client_registry; the teardown pops them back out so no
    other test sees a version "1" (the registry has no deregister API).
    """

    class ProposedFileChangesMessageV1(ProposedFileChangesMessageV0):
        version: str = "1"

    class FileChangeEventsMessageV1(FileChangeEventsMessageV0):
        version: str = "1"

    for canonical_name, v0, v1 in (
        (
            "ProposedFileChangesMessage",
            ProposedFileChangesMessageV0,
            ProposedFileChangesMessageV1,
        ),
        (
            "FileChangeEventsMessage",
            FileChangeEventsMessageV0,
            FileChangeEventsMessageV1,
        ),
    ):
        client_registry.register_migration(
            canonical_name=canonical_name,
            from_version="0",
            to_version="1",
            fn=lambda obj, v1=v1: v1(**obj.model_dump(exclude={"version"})),
        )
        client_registry.register_migration(
            canonical_name=canonical_name,
            from_version="1",
            to_version="0",
            fn=lambda obj, v0=v0: v0(**obj.model_dump(exclude={"version"})),
        )

    yield ProposedFileChangesMessageV1, FileChangeEventsMessageV1

    for canonical_name in ("ProposedFileChangesMessage", "FileChangeEventsMessage"):
        client_registry.objects[canonical_name].pop("1", None)
        client_registry.migrations.get(canonical_name, {}).pop(("0", "1"), None)
        client_registry.migrations.get(canonical_name, {}).pop(("1", "0"), None)


def _wire_version(manager, peer_email: str, raw: bytes) -> str:
    """The unparsed version field of one blob `peer_email` sent to `manager`.

    Decrypts first, so the reader works for an encrypted pair as well as a
    plaintext one. `decrypt_and_verify_if_needed` passes plaintext through, and
    the assert keeps the encrypted parametrization from passing on plaintext.
    """
    peer_store = manager.peer_manager.peer_store
    opened = peer_store.decrypt_and_verify_if_needed(peer_email, raw)
    if peer_store.pair_encrypts(peer_store.get_cached_peer(peer_email)):
        assert opened != raw, "encryption is on, but the wire holds plaintext"
    return json.loads(uncompress_data(opened))["version"]


def _raw_proposal_version(do_manager, ds_email: str) -> str:
    """The version field of the next proposal blob in the DO's inbox, unparsed."""
    [message] = do_manager._connection_router.connections.p2p_con.receive(
        ds_email, MessageKind.PROPOSED_CHANGES, max_messages=1
    )
    return _wire_version(do_manager, ds_email, message.data)


def _raw_outbox_versions(ds_manager, do_email: str) -> list[str]:
    """The version fields of the DO's outbox blobs for us, unparsed."""
    p2p = ds_manager._connection_router.connections.p2p_con
    received = p2p.receive(do_email, MessageKind.FILE_EVENTS)
    return [_wire_version(ds_manager, do_email, m.data) for m in received]


def test_a_v1_proposal_for_a_protocol0_peer_downgrades_on_the_wire(
    pair, v1_wire_envelopes, advertised_schemas
):
    ds_manager, do_manager = pair
    v1_proposed, _ = v1_wire_envelopes
    # The DO advertises client protocol 0, which knows message version 0 only.
    advertised_schemas[do_manager.email, "syft"] = _client_schema("0")

    message = v1_proposed(**mock_message().model_dump(exclude={"version"}))
    ds_manager._connection_router.watcher_send_proposed_file_changes_message(
        do_manager.email, message
    )

    assert _raw_proposal_version(do_manager, ds_manager.email) == "0", (
        "the peer's protocol supports message version 0 only, so the sender "
        "must downgrade before the blob goes up"
    )


def test_a_v1_events_message_for_a_protocol0_peer_downgrades_on_the_wire(
    pair, v1_wire_envelopes, advertised_schemas
):
    ds_manager, do_manager = pair
    _, v1_events = v1_wire_envelopes
    advertised_schemas[ds_manager.email, "syft"] = _client_schema("0")

    message = v1_events(
        **get_mock_events_messages(1)[0].model_dump(exclude={"version"})
    )
    do_manager._connection_router.owner_write_event_messages_to_outbox(
        ds_manager.email, message
    )

    assert _raw_outbox_versions(ds_manager, do_manager.email) == ["0"]


def test_a_send_beyond_the_peers_floor_raises(pair, advertised_schemas):
    # A future peer that dropped support for our protocol. Sending anyway would
    # put up a blob the peer refuses; the negotiation must fail loudly instead.
    ds_manager, do_manager = pair
    advertised_schemas[do_manager.email, "syft"] = _client_schema(
        "1", min_supported_version="1"
    )

    with pytest.raises(MigrationError):
        ds_manager._connection_router.watcher_send_proposed_file_changes_message(
            do_manager.email, mock_message()
        )


def test_a_send_to_an_unknown_peer_warns_and_keeps_the_current_version(
    pair, caplog, advertised_schemas
):
    # Same policy as jobs: a peer without a known schema is assumed to run the
    # current protocol, and the assumption is logged.
    ds_manager, do_manager = pair
    advertised_schemas[do_manager.email, "syft"] = None

    with caplog.at_level(logging.WARNING):
        ds_manager._connection_router.watcher_send_proposed_file_changes_message(
            do_manager.email, mock_message()
        )

    assert "No syft protocol schema known" in caplog.text
    assert _raw_proposal_version(do_manager, ds_manager.email) == "0"


def test_a_current_protocol_peer_gets_the_current_version(pair, advertised_schemas):
    # The control: a peer on our own protocol gets the current version, so the
    # tests above measure the downgrade and not a broken default.
    ds_manager, do_manager = pair
    advertised_schemas[do_manager.email, "syft"] = _client_schema("0")

    ds_manager._connection_router.watcher_send_proposed_file_changes_message(
        do_manager.email, mock_message()
    )

    assert _raw_proposal_version(do_manager, ds_manager.email) == "0"
