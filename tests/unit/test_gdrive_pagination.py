from unittest.mock import Mock, patch
from uuid import uuid4
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_client.sync.events.file_change_event import (
    FileChangeEventsMessage,
    FileChangeEventsMessageFileName,
)


def create_mock_connection():
    conn = GDriveConnection(email="test@test.com", verbose=False)
    conn.drive_service = Mock()
    return conn


def test_single_page_works():
    """Current implementation works for single page (< 100 files)"""
    conn = create_mock_connection()

    conn.drive_service.files().list().execute.return_value = {
        "files": [{"name": "file1.txt", "id": "id1"}]
    }

    files = conn.get_file_metadatas_from_folder("folder_id")

    assert len(files) == 1
    assert files[0]["name"] == "file1.txt"


def test_multiple_pages_pagination():
    """When there are >100 files, should fetch all pages"""
    conn = create_mock_connection()

    conn.drive_service.files().list().execute.side_effect = [
        {"files": [{"name": "file1.txt", "id": "id1"}], "nextPageToken": "token1"},
        {"files": [{"name": "file2.txt", "id": "id2"}], "nextPageToken": "token2"},
        {"files": [{"name": "file3.txt", "id": "id3"}]},
    ]

    files = conn.get_file_metadatas_from_folder("folder_id")

    assert len(files) == 3
    assert conn.drive_service.files().list().execute.call_count == 3


def test_empty_folder():
    """Empty folder should return empty list"""
    conn = create_mock_connection()

    conn.drive_service.files().list().execute.return_value = {"files": []}

    files = conn.get_file_metadatas_from_folder("folder_id")

    assert files == []


@patch(
    "syft_client.sync.connections.drive.gdrive_transport.GDriveConnection._get_peer_datasite_outbox_id"
)
@patch(
    "syft_client.sync.connections.drive.gdrive_transport.GDriveConnection.download_file"
)
def test_events_returned_in_chronological_order(mock_download_file, mock_get_folder_id):
    """Events should be returned in chronological order by timestamp"""
    conn = create_mock_connection()

    # Create valid filenames with proper UUIDs
    uuid1 = uuid4()
    uuid2 = uuid4()
    uuid3 = uuid4()

    # Mock Drive returns files in REVERSE chronological order (3.0, 2.0, 1.0)
    conn.drive_service.files().list().execute.return_value = {
        "files": [
            {"id": "id3", "name": f"syfteventsmessagev3_3.0_{uuid3}.tar.gz"},
            {"id": "id2", "name": f"syfteventsmessagev3_2.0_{uuid2}.tar.gz"},
            {"id": "id1", "name": f"syfteventsmessagev3_1.0_{uuid1}.tar.gz"},
        ]
    }

    # Mock _get_peer_datasite_outbox_id
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

    events = conn.watcher_get_events_messages("peer@test.com", None)

    # Should return in chronological order (1.0, 2.0, 3.0)
    timestamps = [e.timestamp for e in events]
    assert timestamps == [
        1.0,
        2.0,
        3.0,
    ], f"Got {timestamps}, expected [1.0, 2.0, 3.0]"


def test_orderby_name_desc_applied():
    """Drive API should be called with orderBy='name desc'"""
    conn = create_mock_connection()

    conn.drive_service.files().list().execute.return_value = {"files": []}

    conn.get_file_metadatas_from_folder("folder_id")

    # Check that orderBy was passed to Drive API
    call_kwargs = conn.drive_service.files().list.call_args[1]
    assert call_kwargs["orderBy"] == "name desc"


def test_extract_timestamp_from_event_filename():
    """Should extract timestamp from event filename"""
    timestamp = GDriveConnection._extract_timestamp_from_filename(
        "syfteventsmessagev3_1731506400.123_uuid.tar.gz"
    )
    assert timestamp == 1731506400.123


def test_extract_timestamp_from_job_filename():
    """Should extract timestamp from job/message filename"""
    timestamp = GDriveConnection._extract_timestamp_from_filename(
        "msgv2_1731506400.123_uuid.tar.gz"
    )
    assert timestamp == 1731506400.123


def test_extract_timestamp_returns_none_for_invalid():
    """Should return None for non-parseable filenames"""
    timestamp = GDriveConnection._extract_timestamp_from_filename("invalid_file.txt")
    assert timestamp is None


def test_early_termination_stops_pagination():
    """Should stop fetching pages when encountering old file"""
    conn = create_mock_connection()

    # Create filenames using FileChangeEventsMessageFileName
    fname1 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506450.0)
    fname2 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506440.0)
    fname3 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506430.0)
    fname4 = FileChangeEventsMessageFileName(
        id=uuid4(), timestamp=1731506400.0
    )  # <= since_timestamp
    fname5 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506390.0)

    # Mock returns 3 potential pages, but should stop after page 2
    conn.drive_service.files().list().execute.side_effect = [
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
                },  # <= since_timestamp, should stop here
            ],
            "nextPageToken": "token2",
        },
        {
            "files": [{"name": fname5.as_string(), "id": "id5"}],
            "nextPageToken": None,
        },
    ]

    files = conn.get_file_metadatas_from_folder(
        "folder_id", since_timestamp=1731506400.0
    )

    # Should stop after 2nd page (when it finds timestamp <= 1731506400.0)
    assert conn.drive_service.files().list().execute.call_count == 2
    # Only files with timestamp > since_timestamp are returned
    assert len(files) == 3  # fname1, fname2, fname3 (fname4 is excluded)
    # Page 3 should NOT be fetched


def test_no_early_termination_when_all_files_new():
    """Should fetch all pages when all files are newer than since_timestamp"""
    conn = create_mock_connection()

    fname1 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506450.0)
    fname2 = FileChangeEventsMessageFileName(id=uuid4(), timestamp=1731506440.0)

    conn.drive_service.files().list().execute.side_effect = [
        {
            "files": [{"name": fname1.as_string(), "id": "id1"}],
            "nextPageToken": "token1",
        },
        {
            "files": [{"name": fname2.as_string(), "id": "id2"}],
            "nextPageToken": None,
        },
    ]

    files = conn.get_file_metadatas_from_folder(
        "folder_id", since_timestamp=1731506400.0
    )

    # Should fetch all pages (both files are > since_timestamp)
    assert conn.drive_service.files().list().execute.call_count == 2
    assert len(files) == 2


def test_no_early_termination_when_since_timestamp_none():
    """Should fetch all pages when since_timestamp is None"""
    conn = create_mock_connection()

    conn.drive_service.files().list().execute.side_effect = [
        {
            "files": [{"name": "file1.txt", "id": "id1"}],
            "nextPageToken": "token1",
        },
        {
            "files": [{"name": "file2.txt", "id": "id2"}],
            "nextPageToken": None,
        },
    ]

    files = conn.get_file_metadatas_from_folder("folder_id", since_timestamp=None)

    # Should fetch all pages (no early termination)
    assert conn.drive_service.files().list().execute.call_count == 2
    assert len(files) == 2
