"""owner_list_all_collections_with_permissions skips only bad names.

The Drive query matches a name prefix, so another tool can return a folder that
this client cannot parse. The listing skips that folder. Every other failure is a
defect, so the listing must raise it.
"""

from unittest.mock import Mock

import pytest

from syft.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX

VALID = f"{DATASET_COLLECTION_PREFIX}_mytag_abc123"
UNPARSEABLE = DATASET_COLLECTION_PREFIX


def _conn(files):
    conn = GDriveConnection(email="alice@example.com", verbose=False)
    conn.drive_service = Mock()
    conn._syftbox_folder_id = "syftbox-id"
    conn.drive_service.files().list().execute.return_value = {"files": files}
    return conn


def test_a_valid_collection_is_returned():
    conn = _conn([{"id": "f1", "name": VALID, "appProperties": {}}])
    got = conn.owner_list_all_collections_with_permissions(DATASET_COLLECTION_PREFIX)
    assert [(c.folder_id, c.tag, c.content_hash) for c in got] == [
        ("f1", "mytag", "abc123")
    ]
    assert got[0].has_any_permission is False


def test_the_any_permission_flag_comes_from_app_properties():
    conn = _conn(
        [
            {
                "id": "f1",
                "name": VALID,
                "appProperties": {"syft_shared_with_any": "true"},
            }
        ]
    )
    got = conn.owner_list_all_collections_with_permissions(DATASET_COLLECTION_PREFIX)
    assert got[0].has_any_permission is True


def test_a_name_the_client_cannot_parse_is_skipped():
    conn = _conn(
        [
            {"id": "bad", "name": UNPARSEABLE, "appProperties": {}},
            {"id": "f1", "name": VALID, "appProperties": {}},
        ]
    )
    got = conn.owner_list_all_collections_with_permissions(DATASET_COLLECTION_PREFIX)
    assert [c.folder_id for c in got] == ["f1"]


def test_a_missing_name_field_raises():
    # A blanket except turned this defect into a collection that disappears
    # without a message.
    conn = _conn([{"id": "f1", "appProperties": {}}])
    with pytest.raises(KeyError):
        conn.owner_list_all_collections_with_permissions(DATASET_COLLECTION_PREFIX)
