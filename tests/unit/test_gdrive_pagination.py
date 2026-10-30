from unittest.mock import Mock, patch
from uuid import uuid4

from syft.sync.connections.drive.drive_collections import DriveCollectionsConnection
from syft.sync.connections.drive.drive_p2p import DriveP2PConnection
from syft.sync.connections.drive.drive_peering import DrivePeeringConnection
from syft.sync.connections.drive.gdrive_client import GDriveClient
from syft.sync.connections.drive.drive_cached_client import DriveCachedClient
from syft.sync.connections.drive.mock_drive_service import (
    MockDriveBackingStore,
    MockDriveService,
)
from syft.sync.connections.types import MessageKind
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
)


def create_mock_cached_client() -> DriveCachedClient:
    return DriveCachedClient(GDriveClient(Mock(), Mock), "test@test.com")


def test_single_page_works():
    """Current implementation works for single page (< 100 files)"""
    cached_client = create_mock_cached_client()

    cached_client.client.drive_service.files().list().execute.return_value = {
        "files": [{"name": "file1.txt", "id": "id1"}]
    }

    files = cached_client.get_file_metadatas_from_folder("folder_id")

    assert len(files) == 1
    assert files[0]["name"] == "file1.txt"


def test_multiple_pages_pagination():
    """When there are >100 files, should fetch all pages"""
    cached_client = create_mock_cached_client()

    cached_client.client.drive_service.files().list().execute.side_effect = [
        {"files": [{"name": "file1.txt", "id": "id1"}], "nextPageToken": "token1"},
        {"files": [{"name": "file2.txt", "id": "id2"}], "nextPageToken": "token2"},
        {"files": [{"name": "file3.txt", "id": "id3"}]},
    ]

    files = cached_client.get_file_metadatas_from_folder("folder_id")

    assert len(files) == 3
    assert cached_client.client.drive_service.files().list().execute.call_count == 3


def test_empty_folder():
    """Empty folder should return empty list"""
    cached_client = create_mock_cached_client()

    cached_client.client.drive_service.files().list().execute.return_value = {
        "files": []
    }

    files = cached_client.get_file_metadatas_from_folder("folder_id")

    assert files == []


@patch(
    "syft.sync.connections.drive.drive_cached_client.DriveCachedClient.get_peer_datasite_outbox_id"
)
@patch("syft.sync.connections.drive.gdrive_client.GDriveClient.download_file")
def test_events_returned_in_chronological_order(mock_download_file, mock_get_folder_id):
    """Events should be returned in chronological order by timestamp"""
    cached_client = create_mock_cached_client()

    # Create valid filenames with proper UUIDs
    uuid1 = uuid4()
    uuid2 = uuid4()
    uuid3 = uuid4()

    # Mock Drive returns files in REVERSE chronological order (3.0, 2.0, 1.0)
    cached_client.client.drive_service.files().list().execute.return_value = {
        "files": [
            {"id": "id3", "name": f"syfteventsmessagev3_3.0_{uuid3}.tar.gz"},
            {"id": "id2", "name": f"syfteventsmessagev3_2.0_{uuid2}.tar.gz"},
            {"id": "id1", "name": f"syfteventsmessagev3_1.0_{uuid1}.tar.gz"},
        ]
    }

    # Mock get_peer_datasite_outbox_id
    mock_get_folder_id.return_value = "folder_id"

    # Mock download_file to return events with matching timestamps
    def mock_download(file_id):
        timestamp_map = {
            "id1": 1.0,
            "id2": 2.0,
            "id3": 3.0,
        }
        timestamp = timestamp_map[file_id]
        fname = FileChangeEventsMessageFileName(id=uuid4(), timestamp=timestamp)
        event_msg = FileChangeEventsMessage(events=[], message_filepath=fname)
        return event_msg.as_compressed_data()

    mock_download_file.side_effect = mock_download

    events = [
        FileChangeEventsMessage.from_compressed_data(m.data)
        for m in DriveP2PConnection(cached_client).receive(
            "peer@test.com", MessageKind.FILE_EVENTS
        )
    ]

    # Should return in chronological order (1.0, 2.0, 3.0)
    timestamps = [e.timestamp for e in events]
    assert timestamps == [
        1.0,
        2.0,
        3.0,
    ], f"Got {timestamps}, expected [1.0, 2.0, 3.0]"


def test_orderby_name_desc_applied():
    """Drive API should be called with orderBy='name desc'"""
    cached_client = create_mock_cached_client()

    cached_client.client.drive_service.files().list().execute.return_value = {
        "files": []
    }

    cached_client.get_file_metadatas_from_folder("folder_id")

    # Check that orderBy was passed to Drive API
    call_kwargs = cached_client.client.drive_service.files().list.call_args[1]
    assert call_kwargs["orderBy"] == "name desc"


def test_early_termination_stops_pagination():
    """Should stop fetching pages when encountering old file"""
    cached_client = create_mock_cached_client()

    # Create filenames using FileChangeEventsMessageFileName
    fname1 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506450.0)
    fname2 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506440.0)
    fname3 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506430.0)
    fname4 = FileChangeEventsMessageFileName(
        id=uuid4(), timestamp=1731506399.0
    )  # one second older than since_timestamp
    fname5 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506390.0)

    # Mock returns 3 potential pages, but should stop after page 2
    cached_client.client.drive_service.files().list().execute.side_effect = [
        {
            "files": [
                {"name": fname1.as_string(), "id": "id1"},
                {"name": fname2.as_string(), "id": "id2"},
            ],
            "nextPageToken": "token1",
        },
        {
            "files": [
                {"name": fname3.as_string(), "id": "id3"},
                {
                    "name": fname4.as_string(),
                    "id": "id4",
                },  # one second older than since_timestamp, should stop here
            ],
            "nextPageToken": "token2",
        },
        {
            "files": [{"name": fname5.as_string(), "id": "id5"}],
            "nextPageToken": None,
        },
    ]

    files = cached_client.get_file_metadatas_from_folder(
        "folder_id", since_timestamp=1731506400.0
    )

    # Should stop after 2nd page (when it finds a timestamp in an earlier second)
    assert cached_client.client.drive_service.files().list().execute.call_count == 2
    # Only files with timestamp > since_timestamp are returned
    assert len(files) == 3  # fname1, fname2, fname3 (fname4 is excluded)
    # Page 3 should NOT be fetched


def test_no_early_termination_when_all_files_new():
    """Should fetch all pages when all files are newer than since_timestamp"""
    cached_client = create_mock_cached_client()

    fname1 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506450.0)
    fname2 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506440.0)

    cached_client.client.drive_service.files().list().execute.side_effect = [
        {
            "files": [{"name": fname1.as_string(), "id": "id1"}],
            "nextPageToken": "token1",
        },
        {
            "files": [{"name": fname2.as_string(), "id": "id2"}],
            "nextPageToken": None,
        },
    ]

    files = cached_client.get_file_metadatas_from_folder(
        "folder_id", since_timestamp=1731506400.0
    )

    # Should fetch all pages (both files are > since_timestamp)
    assert cached_client.client.drive_service.files().list().execute.call_count == 2
    assert len(files) == 2


def test_no_early_termination_when_since_timestamp_none():
    """Should fetch all pages when since_timestamp is None"""
    cached_client = create_mock_cached_client()

    cached_client.client.drive_service.files().list().execute.side_effect = [
        {
            "files": [{"name": "file1.txt", "id": "id1"}],
            "nextPageToken": "token1",
        },
        {
            "files": [{"name": "file2.txt", "id": "id2"}],
            "nextPageToken": None,
        },
    ]

    files = cached_client.get_file_metadatas_from_folder(
        "folder_id", since_timestamp=None
    )

    # Should fetch all pages (no early termination)
    assert cached_client.client.drive_service.files().list().execute.call_count == 2
    assert len(files) == 2


def test_newer_message_listed_after_older_one_in_same_second_not_skipped():
    cached_client = create_mock_cached_client()
    older = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506400.9)
    newer = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506400.95)
    # Drive lists by "name desc": "…400.9_" sorts before "…400.95_".
    names = sorted([older.as_string(), newer.as_string()], reverse=True)
    cached_client.client.drive_service.files().list().execute.return_value = {
        "files": [{"name": name, "id": name} for name in names]
    }

    files = cached_client.get_file_metadatas_from_folder(
        "folder_id", since_timestamp=1731506400.92
    )

    assert [f["name"] for f in files] == [newer.as_string()]


# Three pages of the mock Drive, which returns 100 files per page.
FILE_COUNT = 250


def _mock_cached_clients(*emails: str) -> list[DriveCachedClient]:
    store = MockDriveBackingStore()
    return [
        DriveCachedClient.from_service(e, MockDriveService(store, e)) for e in emails
    ]


def test_collection_listings_read_every_page():
    do_cached_client, ds_cached_client = _mock_cached_clients(
        "do@example.com", "ds@example.com"
    )
    owner, reader = (
        DriveCollectionsConnection(do_cached_client),
        DriveCollectionsConnection(ds_cached_client),
    )
    tags = [f"tag{i}" for i in range(FILE_COUNT)]
    for tag in tags:
        owner.create("syft_datasetcollection", tag, "hash", "do@example.com")
        owner.grant_read("syft_datasetcollection", tag, "hash", ["ds@example.com"])

    assert sorted(owner.list_mine("syft_datasetcollection")) == sorted(tags)
    mine = owner.list_mine_with_permissions("syft_datasetcollection")
    assert len(mine) == FILE_COUNT
    shared = reader.list_shared_with_me("syft_datasetcollection")
    assert len(shared) == FILE_COUNT


def test_peer_requests_read_every_page():
    peer_emails = [f"peer{i}@example.com" for i in range(FILE_COUNT)]
    do_cached_client, *peers = _mock_cached_clients("do@example.com", *peer_emails)
    for peer in peers:
        DrivePeeringConnection(peer).open_channels("do@example.com")

    requests = DrivePeeringConnection(do_cached_client).incoming_requests()
    assert sorted(requests) == sorted(peer_emails)
