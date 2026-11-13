"""The events envelope round-trips, and a blob without identity fields is refused."""

import json
from uuid import uuid4

import pytest
from syft.migrations import client_registry
from syft.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
    FileChangeEventsMessageV0,
    FileChangeEventV0,
)
from syft.sync.messages.proposed_filechange import ProposedFileChange
from syft.sync.utils.syftbox_utils import compress_data
from syft_migration import MigrationError


def _make_event(content) -> FileChangeEvent:
    return FileChangeEvent(
        id=uuid4(),
        path_in_datasite="data/file.txt",
        datasite_email="do@test.org",
        content=content,
        submitted_timestamp=1752900000.0,
        timestamp=1752900001.0,
    )


def test_envelope_registered_items_not():
    assert client_registry.versions("FileChangeEventsMessage")
    assert not client_registry.versions("FileChangeEvent")
    assert FileChangeEventsMessage is FileChangeEventsMessageV0
    assert FileChangeEvent is FileChangeEventV0


def test_round_trip_text_binary_and_deletion():
    for content in ["hello", b"\x00\x01binary", None]:
        original = FileChangeEventsMessage(events=[_make_event(content)])
        restored = FileChangeEventsMessage.from_compressed_data(
            original.as_compressed_data()
        )
        assert restored.events[0].content == content
        assert restored.events[0].content_type == original.events[0].content_type
        assert restored.message_filepath == original.message_filepath


def test_identity_fields_on_the_wire():
    # load_as_latest would setdefault them back, so the round-trip alone
    # cannot catch a silently broken serialization of the identity fields.
    data = json.loads(
        FileChangeEventsMessage(events=[_make_event("x")]).model_dump_json()
    )
    assert data["canonical_name"] == "FileChangeEventsMessage"
    assert data["version"] == "0"


def test_a_blob_without_identity_fields_is_refused():
    message = FileChangeEventsMessage(events=[_make_event("hello")])
    data = json.loads(message.model_dump_json())
    del data["canonical_name"], data["version"]
    blob = compress_data(json.dumps(data).encode("utf-8"))

    with pytest.raises(MigrationError):
        FileChangeEventsMessage.from_compressed_data(blob)


def test_from_proposed_filechange_carries_identity_free_items():
    proposed = ProposedFileChange(
        path_in_datasite="data/file.txt",
        content="hello",
        datasite_email="do@test.org",
    )
    event = FileChangeEvent.from_proposed_filechange(proposed)
    assert event.id == proposed.id
    assert event.new_hash == proposed.new_hash
    # Items carry no identity fields on the wire; only envelopes do.
    assert "canonical_name" not in json.loads(event.model_dump_json())


def test_from_proposed_filechange_keeps_delete():
    proposed = ProposedFileChange(
        path_in_datasite="data/file.txt",
        old_hash="h1",
        datasite_email="do@test.org",
        is_deleted=True,
    )
    event = FileChangeEvent.from_proposed_filechange(proposed)
    assert event.is_deleted
    assert event.content is None
