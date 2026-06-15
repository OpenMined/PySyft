import time
from concurrent.futures import ThreadPoolExecutor, wait
import io
from syft_client.sync.syftbox_manager import SyftboxManager
import os
from syft_client.sync.events.file_change_event import FileChangeEventsMessage
from pathlib import Path
import httplib2
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Timeout for Google API requests (in seconds)
GOOGLE_API_TIMEOUT = 120  # 2 minutes

SYFT_CLIENT_DIR = Path(__file__).parent.parent
# These are in gitignore, create yourself
CREDENTIALS_DIR = SYFT_CLIENT_DIR / "credentials"

# koen gmail
FILE_DO = os.environ.get("ai_audit_credentials_fname_do", "token_do.json")
EMAIL_DO = os.environ["AI_AUDIT_EMAIL_DO"]

# koen openmined mail
FILE_DS = os.environ.get("ai_audit_credentials_fname_ds", "token_ds.json")
EMAIL_DS = os.environ["AI_AUDIT_EMAIL_DS"]


token_path_do = CREDENTIALS_DIR / FILE_DO
token_path_ds = CREDENTIALS_DIR / FILE_DS


def benchmark_gdrive_load_state():
    manager_ds1, manager_do1 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
            add_peers=False,
            load_peers=True,
        )
    )

    connection = manager_do1._connection_router.connections[0]

    personal_syftbox_folder_id = connection.get_personal_syftbox_folder_id()
    file_metadatas = connection.get_file_metadatas_from_folder(
        personal_syftbox_folder_id
    )
    # valid_fname_objs = connection._get_valid_events_from_file_metadatas(file_metadatas)
    gdrive_ids = [f["id"] for f in file_metadatas]

    def download_file(gdrive_id: str) -> bytes:
        # Create Http with timeout to prevent indefinite hangs
        http = httplib2.Http(timeout=GOOGLE_API_TIMEOUT)
        authorized_http = AuthorizedHttp(connection.credentials, http=http)
        service = build(
            "drive",
            "v3",
            http=authorized_http,
        )
        request = service.files().get_media(fileId=gdrive_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return file_buffer.getvalue()

    print("initial sync")
    start = time.time()

    result = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_file, gdrive_id) for gdrive_id in gdrive_ids
        ]

    results = wait(futures)
    for future in results.done:
        file_data = future.result()
        event = FileChangeEventsMessage.from_compressed_data(file_data)
        result.append(event)

    for res in result:
        print(res.events[0].path_in_datasite)

    end = time.time()

    print(f"Time taken to download {len(result)} files: {end - start} seconds")


if __name__ == "__main__":
    benchmark_gdrive_load_state()
