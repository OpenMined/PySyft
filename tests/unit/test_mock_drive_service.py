"""Tests for MockDriveService and pair_with_mock_drive_service_connection.

These tests verify that the mock Google Drive service provides a compatible
interface for testing GDriveConnection code paths without real API calls.
"""

from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.connections.drive.mock_drive_service import (
    MockDriveBackingStore,
    MockDriveService,
    MockDriveFile,
    MockPermission,
    parse_gdrive_query,
    GOOGLE_FOLDER_MIME_TYPE,
)
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from tests.unit.utils import create_tmp_dataset_files


class TestMockDriveBackingStore:
    """Tests for MockDriveBackingStore."""

    def test_add_and_get_file(self):
        store = MockDriveBackingStore()
        file = MockDriveFile(
            name="test.txt",
            content=b"Hello, world!",
            owners=[{"emailAddress": "user@example.com"}],
        )
        store.add_file(file)

        retrieved = store.get_file(file.id)
        assert retrieved is not None
        assert retrieved.name == "test.txt"
        assert retrieved.content == b"Hello, world!"

    def test_delete_file(self):
        store = MockDriveBackingStore()
        file = MockDriveFile(name="test.txt")
        store.add_file(file)

        store.delete_file(file.id)
        assert store.get_file(file.id) is None

    def test_permissions(self):
        store = MockDriveBackingStore()
        file = MockDriveFile(
            name="test.txt",
            owners=[{"emailAddress": "owner@example.com"}],
        )
        store.add_file(file)

        # Owner permission is added automatically
        perms = store.get_permissions(file.id)
        assert len(perms) == 1
        assert perms[0].role == "owner"

        # Add another permission
        store.add_permission(
            file.id,
            MockPermission(
                type="user", role="reader", emailAddress="reader@example.com"
            ),
        )
        perms = store.get_permissions(file.id)
        assert len(perms) == 2


class TestQueryParser:
    """Tests for parse_gdrive_query."""

    def test_empty_query(self):
        filter_func = parse_gdrive_query("", "user@example.com")
        file = MockDriveFile(name="test.txt")
        assert filter_func(file) is True

    def test_name_equals(self):
        filter_func = parse_gdrive_query("name='test.txt'", "user@example.com")
        assert filter_func(MockDriveFile(name="test.txt")) is True
        assert filter_func(MockDriveFile(name="other.txt")) is False

    def test_name_contains(self):
        filter_func = parse_gdrive_query("name contains 'test'", "user@example.com")
        assert filter_func(MockDriveFile(name="my_test_file.txt")) is True
        assert filter_func(MockDriveFile(name="other.txt")) is False

    def test_mime_type(self):
        filter_func = parse_gdrive_query(
            f"mimeType='{GOOGLE_FOLDER_MIME_TYPE}'", "user@example.com"
        )
        assert (
            filter_func(MockDriveFile(name="folder", mimeType=GOOGLE_FOLDER_MIME_TYPE))
            is True
        )
        assert (
            filter_func(MockDriveFile(name="file.txt", mimeType="text/plain")) is False
        )

    def test_in_parents(self):
        filter_func = parse_gdrive_query("'parent123' in parents", "user@example.com")
        assert (
            filter_func(MockDriveFile(name="test.txt", parents=["parent123"])) is True
        )
        assert filter_func(MockDriveFile(name="test.txt", parents=["other"])) is False

    def test_me_in_owners(self):
        filter_func = parse_gdrive_query("'me' in owners", "user@example.com")
        assert (
            filter_func(
                MockDriveFile(
                    name="test.txt", owners=[{"emailAddress": "user@example.com"}]
                )
            )
            is True
        )
        assert (
            filter_func(
                MockDriveFile(
                    name="test.txt", owners=[{"emailAddress": "other@example.com"}]
                )
            )
            is False
        )

    def test_not_me_in_owners(self):
        filter_func = parse_gdrive_query("not 'me' in owners", "user@example.com")
        assert (
            filter_func(
                MockDriveFile(
                    name="test.txt", owners=[{"emailAddress": "other@example.com"}]
                )
            )
            is True
        )
        assert (
            filter_func(
                MockDriveFile(
                    name="test.txt", owners=[{"emailAddress": "user@example.com"}]
                )
            )
            is False
        )

    def test_email_in_owners(self):
        filter_func = parse_gdrive_query(
            "'owner@example.com' in owners", "user@example.com"
        )
        assert (
            filter_func(
                MockDriveFile(
                    name="test.txt", owners=[{"emailAddress": "owner@example.com"}]
                )
            )
            is True
        )
        assert (
            filter_func(
                MockDriveFile(
                    name="test.txt", owners=[{"emailAddress": "other@example.com"}]
                )
            )
            is False
        )

    def test_trashed_false(self):
        filter_func = parse_gdrive_query("trashed=false", "user@example.com")
        assert filter_func(MockDriveFile(name="test.txt", trashed=False)) is True
        assert filter_func(MockDriveFile(name="test.txt", trashed=True)) is False

    def test_combined_query(self):
        filter_func = parse_gdrive_query(
            "name='test.txt' and 'me' in owners and trashed=false",
            "user@example.com",
        )
        file_match = MockDriveFile(
            name="test.txt",
            owners=[{"emailAddress": "user@example.com"}],
            trashed=False,
        )
        file_no_match = MockDriveFile(
            name="other.txt",
            owners=[{"emailAddress": "user@example.com"}],
            trashed=False,
        )
        assert filter_func(file_match) is True
        assert filter_func(file_no_match) is False


class TestMockDriveService:
    """Tests for MockDriveService."""

    def test_create_and_list_files(self):
        store = MockDriveBackingStore()
        service = MockDriveService(store, "user@example.com")

        # Create a file
        result = (
            service.files()
            .create(
                body={"name": "test.txt"},
                media_body=None,
                fields="id,name",
            )
            .execute()
        )

        assert "id" in result
        assert result["name"] == "test.txt"

        # List files
        list_result = (
            service.files()
            .list(
                q="name='test.txt' and 'me' in owners and trashed=false",
                fields="files(id,name)",
            )
            .execute()
        )

        assert len(list_result["files"]) == 1
        assert list_result["files"][0]["name"] == "test.txt"

    def test_create_folder(self):
        store = MockDriveBackingStore()
        service = MockDriveService(store, "user@example.com")

        result = (
            service.files()
            .create(
                body={
                    "name": "MyFolder",
                    "mimeType": GOOGLE_FOLDER_MIME_TYPE,
                },
                fields="id",
            )
            .execute()
        )

        assert "id" in result

        # List folders
        list_result = (
            service.files()
            .list(
                q=f"mimeType='{GOOGLE_FOLDER_MIME_TYPE}' and 'me' in owners and trashed=false",
                fields="files(id,name)",
            )
            .execute()
        )

        assert len(list_result["files"]) == 1
        assert list_result["files"][0]["name"] == "MyFolder"

    def test_update_file(self):
        store = MockDriveBackingStore()
        service = MockDriveService(store, "user@example.com")

        # Create a file
        file = (
            service.files()
            .create(
                body={"name": "test.txt"},
                fields="id",
            )
            .execute()
        )

        # Create a folder
        folder = (
            service.files()
            .create(
                body={"name": "MyFolder", "mimeType": GOOGLE_FOLDER_MIME_TYPE},
                fields="id",
            )
            .execute()
        )

        # Update file to add parent
        service.files().update(
            fileId=file["id"],
            addParents=folder["id"],
            fields="id,parents",
        ).execute()

        # Verify parent was added
        list_result = (
            service.files()
            .list(
                q=f"'{folder['id']}' in parents and trashed=false",
                fields="files(id,name)",
            )
            .execute()
        )

        assert len(list_result["files"]) == 1

    def test_delete_file(self):
        store = MockDriveBackingStore()
        service = MockDriveService(store, "user@example.com")

        # Create a file
        file = (
            service.files()
            .create(
                body={"name": "test.txt"},
                fields="id",
            )
            .execute()
        )

        # Delete the file
        service.files().delete(fileId=file["id"]).execute()

        # Verify file is gone
        list_result = (
            service.files()
            .list(
                q="name='test.txt' and 'me' in owners and trashed=false",
                fields="files(id)",
            )
            .execute()
        )

        assert len(list_result["files"]) == 0

    def test_get_media(self):
        store = MockDriveBackingStore()
        service = MockDriveService(store, "user@example.com")

        # Add a file with content directly to the store
        file = MockDriveFile(
            name="test.txt",
            content=b"Hello, world!",
            owners=[{"emailAddress": "user@example.com"}],
        )
        store.add_file(file)

        # Get the content
        content = service.files().get_media(fileId=file.id).execute()
        assert content == b"Hello, world!"

    def test_permissions(self):
        store = MockDriveBackingStore()
        service = MockDriveService(store, "user@example.com")

        # Create a file
        file = (
            service.files()
            .create(
                body={"name": "test.txt"},
                fields="id",
            )
            .execute()
        )

        # Add permission
        perm_result = (
            service.permissions()
            .create(
                fileId=file["id"],
                body={
                    "type": "user",
                    "role": "reader",
                    "emailAddress": "reader@example.com",
                },
                sendNotificationEmail=False,
            )
            .execute()
        )

        assert "id" in perm_result

        # List permissions
        list_result = (
            service.permissions()
            .list(
                fileId=file["id"],
                fields="permissions(type,role)",
            )
            .execute()
        )

        # Should have owner + reader permissions
        assert len(list_result["permissions"]) == 2

    def test_shared_file_access(self):
        """Test that shared files can be accessed by recipients."""
        store = MockDriveBackingStore()
        owner_service = MockDriveService(store, "owner@example.com")
        reader_service = MockDriveService(store, "reader@example.com")

        # Owner creates a file
        file = (
            owner_service.files()
            .create(
                body={"name": "shared.txt"},
                fields="id",
            )
            .execute()
        )

        # Reader can't see it yet
        list_result = (
            reader_service.files()
            .list(
                q="name='shared.txt'",
                fields="files(id)",
            )
            .execute()
        )
        assert len(list_result["files"]) == 0

        # Owner shares with reader
        owner_service.permissions().create(
            fileId=file["id"],
            body={
                "type": "user",
                "role": "reader",
                "emailAddress": "reader@example.com",
            },
        ).execute()

        # Now reader can see it
        list_result = (
            reader_service.files()
            .list(
                q="name='shared.txt'",
                fields="files(id)",
            )
            .execute()
        )
        assert len(list_result["files"]) == 1


class TestGDriveConnectionWithMock:
    """Tests for GDriveConnection with mock service."""

    def test_from_mock_service(self):
        store = MockDriveBackingStore()
        mock_service = MockDriveService(store, "user@example.com")

        connection = GDriveConnection.from_service("user@example.com", mock_service)

        assert connection.email == "user@example.com"
        assert connection._is_setup is True

    def test_create_folder(self):
        store = MockDriveBackingStore()
        mock_service = MockDriveService(store, "user@example.com")
        connection = GDriveConnection.from_service("user@example.com", mock_service)

        folder_id = connection.create_folder("TestFolder", None)
        assert folder_id is not None

    def test_download_file(self):
        store = MockDriveBackingStore()

        # Add a file to the store
        file = MockDriveFile(
            name="test.txt",
            content=b"Test content",
            owners=[{"emailAddress": "user@example.com"}],
        )
        store.add_file(file)

        mock_service = MockDriveService(store, "user@example.com")
        connection = GDriveConnection.from_service("user@example.com", mock_service)

        content = connection.download_file(file.id)
        assert content == b"Test content"


class TestPairWithMockDriveServiceConnection:
    """Tests for SyftboxManager.pair_with_mock_drive_service_connection."""

    def test_basic_pairing(self):
        ds_manager, do_manager = (
            SyftboxManager.pair_with_mock_drive_service_connection()
        )

        assert ds_manager.email is not None
        assert do_manager.email is not None
        assert ds_manager.email != do_manager.email

    def test_peers_setup(self):
        ds_manager, do_manager = (
            SyftboxManager.pair_with_mock_drive_service_connection()
        )

        # Peers should be set up
        do_manager.load_peers()
        assert ds_manager.email in [
            p.email for p in do_manager.peer_manager.approved_peers
        ]

    def test_ds_to_do_file_sync(self):
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            sync_automatically=False,
        )

        # DS sends a file change to DO's job folder (where DS has write access)
        from tests.unit.test_sync_manager import path_for_job

        file_path = path_for_job(do_manager.email, ds_manager.email)
        ds_manager._send_file_change(file_path, "Hello from DS!")

        # DO syncs to receive the message
        do_manager.sync()

        # Check that DO received the event
        events = do_manager._get_all_accepted_events_do()
        assert len(events) > 0

    def test_dataset_creation_and_sync(self):
        """Test that datasets created by DO are visible to DS via mock drive."""
        ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
            use_in_memory_cache=False,
        )

        mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

        # DO creates a dataset
        do_manager.create_dataset(
            name="mock drive dataset",
            mock_path=mock_dset_path,
            private_path=private_dset_path,
            summary="Test dataset via mock drive",
            readme_path=readme_path,
            tags=["test"],
            users=[ds_manager.email],
        )

        # Verify DO can see it
        do_datasets = do_manager.datasets.get_all()
        assert len(do_datasets) == 1

        # DS syncs
        ds_manager.sync()

        # DS should see the dataset
        ds_datasets = ds_manager.datasets.get_all()
        assert len(ds_datasets) == 1

        dataset = ds_manager.datasets.get(
            "mock drive dataset", datasite=do_manager.email
        )
        assert dataset is not None
        assert len(dataset.mock_files) > 0
