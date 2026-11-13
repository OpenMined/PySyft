"""The msgv2 envelope round-trips, and a blob without identity fields is refused."""

import json

import pytest
from syft.migrations import client_registry
from syft.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
    ProposedFileChangesMessageV0,
    ProposedFileChangeV0,
)
from syft.sync.utils.syftbox_utils import compress_data
from syft_migration import MigrationError


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
    assert ProposedFileChangesMessage is ProposedFileChangesMessageV0
    assert ProposedFileChange is ProposedFileChangeV0


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


def test_a_blob_without_identity_fields_is_refused():
    data = json.loads(_make_message("hello").model_dump_json())
    del data["canonical_name"], data["version"]
    blob = compress_data(json.dumps(data).encode("utf-8"))

    with pytest.raises(MigrationError):
        ProposedFileChangesMessage.from_compressed_data(blob)


def test_identity_fields_on_wire_but_platform_id_excluded():
    data = json.loads(_make_message("x").model_dump_json())
    assert data["canonical_name"] == "ProposedFileChangesMessage"
    assert data["version"] == "0"
    assert "platform_id" not in data
