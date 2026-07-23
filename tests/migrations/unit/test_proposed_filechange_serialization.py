"""The msgv2 envelope round-trips and legacy protocol-0 blobs still decode."""

import base64
import json

from syft_client.migrations import client_registry
from syft_client.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
    ProposedFileChangesMessageV1,
    ProposedFileChangeV1,
)
from syft_client.sync.utils.syftbox_utils import compress_data


def _make_message(content) -> ProposedFileChangesMessage:
    return ProposedFileChangesMessage(
        sender_email="ds@test.org",
        proposed_file_changes=[
            ProposedFileChange(
                path_in_datasite="data/file.txt",
                content=content,
                datasite_email="do@test.org",
            )
        ],
    )


def test_envelope_registered_items_not():
    # The envelope is the migratable unit; items ride inside it.
    assert client_registry.versions("ProposedFileChangesMessage")
    assert not client_registry.versions("ProposedFileChange")
    assert ProposedFileChangesMessage is ProposedFileChangesMessageV1
    assert ProposedFileChange is ProposedFileChangeV1


def test_round_trip_text_and_binary():
    for content in ["hello", b"\x00\x01binary"]:
        original = _make_message(content)
        restored = ProposedFileChangesMessage.from_compressed_data(
            original.as_compressed_data()
        )
        assert restored.sender_email == original.sender_email
        assert restored.proposed_file_changes[0].content == content
        assert (
            restored.proposed_file_changes[0].new_hash
            == original.proposed_file_changes[0].new_hash
        )


def test_legacy_protocol0_blob_decodes_as_latest():
    # A blob exactly as a <= 0.1.117 client writes it: no identity fields
    # on the envelope, base64 binary content on the item.
    legacy = {
        "id": "8be509b2-4340-44db-a3a4-b0ecf8c463f4",
        "sender_email": "ds@test.org",
        "message_filename": {
            "submitted_timestamp": 1752900000.0,
            "uid": "6f9d5f57-31f7-4302-8746-9ba030e88961",
        },
        "proposed_file_changes": [
            {
                "id": "9c1a2e75-8a45-4a17-b7f2-0d94d13d3c60",
                "old_hash": None,
                "submitted_timestamp": 1752900000.0,
                "path_in_datasite": "data/blob.bin",
                "content": base64.b64encode(b"\x00\x01binary").decode("utf-8"),
                "content_type": "binary",
                "datasite_email": "do@test.org",
                "is_deleted": False,
            }
        ],
    }
    blob = compress_data(json.dumps(legacy).encode("utf-8"))

    message = ProposedFileChangesMessage.from_compressed_data(blob)
    assert message.version == client_registry.latest_version(
        "ProposedFileChangesMessage"
    )
    change = message.proposed_file_changes[0]
    assert change.content == b"\x00\x01binary"
    assert change.content_type == "binary"
    # pre_init derived the hash from the payload content only.
    assert change.new_hash


def test_identity_fields_on_wire_but_platform_id_excluded():
    data = json.loads(_make_message("x").model_dump_json())
    assert data["canonical_name"] == "ProposedFileChangesMessage"
    assert data["version"] == "1"
    assert "platform_id" not in data
