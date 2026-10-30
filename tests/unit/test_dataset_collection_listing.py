"""Collections.list_mine_with_permissions on Drive skips only bad names.

The Drive query matches a name prefix, so another tool can return a folder that
this client cannot parse. The listing skips that folder. Every other failure is a
defect, so the listing must raise it.
"""

from unittest.mock import Mock

import pytest
from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX

from syft.sync.connections.drive.drive_collections import DriveCollections
from syft.sync.connections.drive.gdrive_client import GDriveClient
from syft.sync.connections.drive.gdrive_layout import DriveLayout

VALID = f"{DATASET_COLLECTION_PREFIX}_mytag_abc123"
UNPARSEABLE = DATASET_COLLECTION_PREFIX


def _collections(files):
    layout = DriveLayout(GDriveClient(Mock(), Mock), "alice@example.com")
    layout._syftbox_folder_id = "syftbox-id"
    layout.client.drive_service.files().list().execute.return_value = {"files": files}
    return DriveCollections(layout)


def test_a_valid_collection_is_returned():
    collections = _collections([{"id": "f1", "name": VALID, "appProperties": {}}])
    got = collections.list_mine_with_permissions(DATASET_COLLECTION_PREFIX)
    assert [(c.folder_id, c.tag, c.content_hash) for c in got] == [
        ("f1", "mytag", "abc123")
    ]
    assert got[0].has_any_permission is False


def test_the_any_permission_flag_comes_from_app_properties():
    collections = _collections(
        [
            {
                "id": "f1",
                "name": VALID,
                "appProperties": {"syft_shared_with_any": "true"},
            }
        ]
    )
    got = collections.list_mine_with_permissions(DATASET_COLLECTION_PREFIX)
    assert got[0].has_any_permission is True


def test_a_name_the_client_cannot_parse_is_skipped():
    collections = _collections(
        [
            {"id": "bad", "name": UNPARSEABLE, "appProperties": {}},
            {"id": "f1", "name": VALID, "appProperties": {}},
        ]
    )
    got = collections.list_mine_with_permissions(DATASET_COLLECTION_PREFIX)
    assert [c.folder_id for c in got] == ["f1"]


def test_a_missing_name_field_raises():
    # A blanket except turned this defect into a collection that disappears
    # without a message.
    collections = _collections([{"id": "f1", "appProperties": {}}])
    with pytest.raises(KeyError):
        collections.list_mine_with_permissions(DATASET_COLLECTION_PREFIX)
