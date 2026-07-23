"""The events envelope round-trips and legacy protocol-0 blobs still decode."""

import base64
import json
from uuid import uuid4

from syft_client.migrations import client_registry
from syft_client.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
    FileChangeEventsMessageV1,
    FileChangeEventV1,
)
from syft_client.sync.messages.proposed_filechange import ProposedFileChange
from syft_client.sync.utils.syftbox_utils import compress_data


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
    assert FileChangeEventsMessage is FileChangeEventsMessageV1
    assert FileChangeEvent is FileChangeEventV1


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
    assert data["version"] == "1"


def test_legacy_protocol0_blob_decodes_as_latest():
    # A blob exactly as a <= 0.1.117 client writes it: no identity fields.
    legacy = {
        "events": [
            {
                "id": "9c1a2e75-8a45-4a17-b7f2-0d94d13d3c60",
                "path_in_datasite": "data/blob.bin",
                "datasite_email": "do@test.org",
                "content": base64.b64encode(b"\x00\x01binary").decode("utf-8"),
                "content_type": "binary",
                "old_hash": None,
                "new_hash": "abc123",
                "is_deleted": False,
                "submitted_timestamp": 1752900000.0,
                "timestamp": 1752900001.0,
            }
        ],
        "message_filepath": {
            "id": "6f9d5f57-31f7-4302-8746-9ba030e88961",
            "timestamp": 1752900001.0,
            "extension": ".tar.gz",
        },
    }
    blob = compress_data(json.dumps(legacy).encode("utf-8"))

    message = FileChangeEventsMessage.from_compressed_data(blob)
    assert message.version == client_registry.latest_version("FileChangeEventsMessage")
    assert message.events[0].content == b"\x00\x01binary"


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
