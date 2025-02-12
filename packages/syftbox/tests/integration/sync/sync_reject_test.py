from pathlib import Path

from fastapi.testclient import TestClient

from syftbox.client.base import SyftBoxContextInterface
from syftbox.client.plugins.sync.manager import SyncManager
from syftbox.client.plugins.sync.sync_action import format_rejected_path
from syftbox.client.utils.dir_tree import create_dir_tree
from syftbox.lib.constants import PERM_FILE
from syftbox.lib.permissions import SyftPermission


def test_create_without_permission(
    server_client: TestClient, datasite_1: SyftBoxContextInterface, datasite_2: SyftBoxContextInterface
):
    # server_settings: ServerSettings = server_client.app_state["server_settings"]
    sync_service_1 = SyncManager(datasite_1)
    sync_service_2 = SyncManager(datasite_2)

    # Create a folder with only read permission for datasite_2
    tree = {
        "folder_1": {
            PERM_FILE: SyftPermission.mine_with_public_read(datasite_1, dir=datasite_1.my_datasite / "folder_1"),
        },
    }
    create_dir_tree(Path(datasite_1.my_datasite), tree)

    sync_service_1.run_single_thread()
    sync_service_2.run_single_thread()

    folder_on_ds1 = datasite_1.workspace.datasites / datasite_1.email / "folder_1"
    folder_on_ds2 = datasite_2.workspace.datasites / datasite_1.email / "folder_1"

    # check if datasite_1/folder_1/PERM_FILE exists on datasite_2
    assert (folder_on_ds2 / PERM_FILE).exists()

    # create a file in folder_1 and sync
    new_file = folder_on_ds2 / "file.txt"
    new_file.write_text("Hello, World!")

    # TODO server currently does not return 403, but unhandled exception
    sync_service_2.run_single_thread()
    sync_service_1.run_single_thread()

    # creating file.txt has been rejected
    assert not (folder_on_ds1 / "file.txt").exists()
    assert not (folder_on_ds2 / "file.txt").exists()
    assert format_rejected_path(folder_on_ds2 / "file.txt").exists()

    # rejected file does not get synced
    sync_service_2.run_single_thread()
    sync_service_1.run_single_thread()
    assert not (folder_on_ds1 / "file.txt.rejected").exists()


def test_delete_without_permission(
    server_client: TestClient, datasite_1: SyftBoxContextInterface, datasite_2: SyftBoxContextInterface
):
    # server_settings: ServerSettings = server_client.app_state["server_settings"]
    sync_service_1 = SyncManager(datasite_1)
    sync_service_2 = SyncManager(datasite_2)

    # Create a folder with only read permission for datasite_2
    tree = {
        "folder_1": {
            PERM_FILE: SyftPermission.mine_with_public_read(datasite_1, dir=datasite_1.my_datasite / "folder_1"),
            "file.txt": "Hello, World!",
        },
    }
    create_dir_tree(Path(datasite_1.my_datasite), tree)

    sync_service_1.run_single_thread()
    sync_service_2.run_single_thread()

    folder_on_ds1 = datasite_1.workspace.datasites / datasite_1.email / "folder_1"
    folder_on_ds2 = datasite_2.workspace.datasites / datasite_1.email / "folder_1"

    # Delete file.txt on datasite_2 is rejected
    (folder_on_ds2 / "file.txt").unlink(missing_ok=False)
    sync_service_2.run_single_thread()
    sync_service_1.run_single_thread()

    assert (folder_on_ds1 / "file.txt").exists()
    assert (folder_on_ds2 / "file.txt").exists()


def test_modify_without_permissions(
    server_client: TestClient, datasite_1: SyftBoxContextInterface, datasite_2: SyftBoxContextInterface
):
    # server_settings: ServerSettings = server_client.app_state["server_settings"]
    sync_service_1 = SyncManager(datasite_1)
    sync_service_2 = SyncManager(datasite_2)

    # Create a folder with only read permission for datasite_2
    tree = {
        "folder_1": {
            PERM_FILE: SyftPermission.mine_with_public_read(datasite_1, dir=datasite_1.my_datasite / "folder_1"),
            "file.txt": "Hello, World!",
        },
    }
    create_dir_tree(Path(datasite_1.my_datasite), tree)

    sync_service_1.run_single_thread()
    sync_service_2.run_single_thread()

    folder_on_ds1 = datasite_1.workspace.datasites / datasite_1.email / "folder_1"
    folder_on_ds2 = datasite_2.workspace.datasites / datasite_1.email / "folder_1"

    # Modify file.txt on datasite_2 is rejected
    (folder_on_ds2 / "file.txt").write_text("Modified")
    sync_service_2.run_single_thread()
    sync_service_1.run_single_thread()

    assert (folder_on_ds1 / "file.txt").read_text() == "Hello, World!"
    assert (folder_on_ds2 / "file.txt").read_text() == "Hello, World!"
    assert format_rejected_path(folder_on_ds2 / "file.txt").read_text() == "Modified"
