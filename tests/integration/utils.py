from syft_client.sync.events.file_change_event import FileChangeEvent
from syft_client.sync.events.file_change_event import FileChangeEventsMessageFileName
import os
import uuid
import time
from pathlib import Path
from typing import List
from syft_client.sync.utils.syftbox_utils import get_event_hash_from_content
from syft_client.sync.syftbox_manager import SyftboxManager

SYFT_CLIENT_DIR = Path(__file__).parent.parent.parent
CREDENTIALS_DIR = SYFT_CLIENT_DIR / "credentials"

FILE_DO = os.environ.get("ai_audit_credentials_fname_do", "token_do.json")
EMAIL_DO = os.environ.get("AI_AUDIT_EMAIL_DO", "")

FILE_DS = os.environ.get("ai_audit_credentials_fname_ds", "token_ds.json")
EMAIL_DS = os.environ.get("AI_AUDIT_EMAIL_DS", "")

token_path_do = CREDENTIALS_DIR / FILE_DO
token_path_ds = CREDENTIALS_DIR / FILE_DS


def remove_syftboxes_from_drive():
    manager_ds, manager_do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=False,
    )
    manager_ds.delete_syftbox(broadcast_delete_events=False)
    manager_do.delete_syftbox(broadcast_delete_events=False)


def get_mock_event(path: str = "email@email.com/test.job") -> FileChangeEvent:
    email = path.split("/")[0]
    file_path = path.split("/")[-1]
    content = "Hello, world!"
    new_hash = get_event_hash_from_content(content)
    return FileChangeEvent(
        datasite_email=email,
        id=uuid.uuid4(),
        path_in_datasite=file_path,
        submitted_timestamp=time.time(),
        timestamp=time.time(),
        content=content,
        new_hash=new_hash,
        event_filepath=FileChangeEventsMessageFileName(
            id=uuid.uuid4(),
            file_path_in_datasite=file_path,
            timestamp=time.time(),
        ),
    )


def get_mock_events(email: str, n_events: int = 2) -> List[FileChangeEvent]:
    res = [get_mock_event(f"{email}/test{i}.job") for i in range(n_events)]
    return res
