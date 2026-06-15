"""
Reproduce the partial delete_syftbox issue.

The bug: after create_dataset + delete_syftbox, local state (private/syft_datasets,
public/syft_datasets) is NOT cleaned up, which causes issues when re-creating
datasets with the same name.

Usage:
    # Step 1: Create data and delete syftboxes, then inspect local + GDrive state
    uv run python scripts/reproduce_delete_issue.py --step setup

    # Step 2: Query GDrive API for leftover files
    uv run python scripts/reproduce_delete_issue.py --step query
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from syft_client import SYFT_CLIENT_DIR
from syft_client.sync.syftbox_manager import SyftboxManager

CREDENTIALS_DIR = SYFT_CLIENT_DIR / "credentials"
SCOPES = ["https://www.googleapis.com/auth/drive"]

FILE_DO = os.environ.get("ai_audit_credentials_fname_do", "token_do.json")
EMAIL_DO = os.environ["AI_AUDIT_EMAIL_DO"]
FILE_DS = os.environ.get("ai_audit_credentials_fname_ds", "token_ds.json")
EMAIL_DS = os.environ["AI_AUDIT_EMAIL_DS"]

token_path_do = CREDENTIALS_DIR / FILE_DO
token_path_ds = CREDENTIALS_DIR / FILE_DS

# Use fixed local paths to simulate VM behavior (persistent folder)
FIXED_DO_PATH = Path(tempfile.gettempdir()) / "repro_delete_DO"
FIXED_DS_PATH = Path(tempfile.gettempdir()) / "repro_delete_DS"

if not token_path_do.exists() or not token_path_ds.exists():
    raise ValueError(
        "Credentials not found. Create them using scripts/create_token.py "
        "and store in /credentials as token_do.json and token_ds.json."
    )


def _execute_with_retry(request, retries=3, delay=2):
    """Execute a GDrive API request with retries for transient errors."""
    for attempt in range(retries):
        try:
            return request.execute()
        except Exception as e:
            if attempt < retries - 1 and ("500" in str(e) or "Internal" in str(e)):
                print(f"  Retry {attempt + 1}/{retries} after error: {e}")
                time.sleep(delay)
            else:
                raise


def query_syftbox_on_gdrive(token_path: Path, label: str):
    """Query Google Drive for any SyftBox-related files/folders."""
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    service = build("drive", "v3", credentials=creds)

    print(f"\n{'=' * 60}")
    print(f"GDrive state for: {label}")
    print(f"{'=' * 60}")

    query = "name='SyftBox' and mimeType='application/vnd.google-apps.folder' and 'me' in owners and trashed=false"
    results = _execute_with_retry(
        service.files().list(q=query, fields="files(id, name)")
    )
    syftbox_folders = results.get("files", [])

    if not syftbox_folders:
        print("  No SyftBox folder found on GDrive.")
        return

    for sf in syftbox_folders:
        print(f"\n  SyftBox folder: id={sf['id']}")
        _list_children(service, sf["id"], indent=4)

    # Search for orphaned syft files
    for pattern in ["syft_datasetcollection", "syft_privatecollection"]:
        query = f"name contains '{pattern}' and 'me' in owners and trashed=false"
        results = _execute_with_retry(
            service.files().list(q=query, fields="files(id, name, mimeType, parents)")
        )
        files = results.get("files", [])
        if files:
            print(f"\n  Orphaned '{pattern}' files:")
            for f in files:
                ftype = "FOLDER" if "folder" in f.get("mimeType", "") else "FILE"
                print(f"    [{ftype}] {f['name']} (id={f['id']})")


def _list_children(service, folder_id, indent=2):
    """Recursively list children of a GDrive folder."""
    query = f"'{folder_id}' in parents and trashed=false"
    results = _execute_with_retry(
        service.files().list(q=query, fields="files(id, name, mimeType)")
    )
    children = results.get("files", [])

    if not children:
        print(f"{' ' * indent}(empty)")
        return

    for child in children:
        is_folder = "folder" in child.get("mimeType", "")
        marker = "DIR " if is_folder else "FILE"
        print(f"{' ' * indent}[{marker}] {child['name']} (id={child['id']})")
        if is_folder:
            _list_children(service, child["id"], indent + 4)


def _show_local_folder(folder: Path, label: str):
    """Show contents of a local syftbox folder."""
    print(f"\n  Local {label} folder: {folder}")
    if not folder.exists():
        print("    (does not exist)")
        return
    for item in sorted(folder.rglob("*")):
        rel = item.relative_to(folder)
        marker = "DIR " if item.is_dir() else "FILE"
        print(f"    [{marker}] {rel}")


def _create_pair(add_peers=True, load_peers=False):
    """Create a DO/DS pair using fixed local paths (simulating persistent VM folders)."""
    return SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        base_path1=str(FIXED_DO_PATH),
        base_path2=str(FIXED_DS_PATH),
        add_peers=add_peers,
        load_peers=load_peers,
    )


def step_setup():
    """Create a dataset, then delete syftboxes to reproduce partial deletion."""

    print("Step 1: Clean up any existing state...")
    manager_ds, manager_do = _create_pair(add_peers=False, load_peers=False)
    manager_do.delete_syftbox(broadcast_delete_events=False)
    manager_ds.delete_syftbox(broadcast_delete_events=False)
    # Also clean local dirs for a fresh start
    import shutil

    if FIXED_DO_PATH.exists():
        shutil.rmtree(FIXED_DO_PATH)
    if FIXED_DS_PATH.exists():
        shutil.rmtree(FIXED_DS_PATH)

    print("\nStep 2: Create managers with peer approval (fixed local paths)...")
    manager_ds, manager_do = _create_pair(add_peers=True)
    print(f"  DO local: {manager_do.syftbox_folder}")
    print(f"  DS local: {manager_ds.syftbox_folder}")

    print("\nStep 3: Create a test dataset with mock + private data...")
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = Path(tmpdir) / "mock_data.csv"
        mock_file.write_text("id,value\n1,hello\n2,world\n")
        private_file = Path(tmpdir) / "private_data.csv"
        private_file.write_text("id,secret\n1,password123\n2,secret456\n")

        dataset = manager_do.create_dataset(
            name="test_delete_issue",
            summary="Testing delete_syftbox partial deletion",
            mock_path=str(mock_file),
            private_path=str(private_file),
            upload_private=True,
            users=[EMAIL_DS],
            sync=True,
        )
        print(f"  Created dataset: {dataset.name}")

    print("\nStep 4: Show state BEFORE delete...")
    _show_local_folder(FIXED_DO_PATH, "DO")
    query_syftbox_on_gdrive(token_path_do, f"DO ({EMAIL_DO}) - BEFORE delete")

    print("\n\nStep 5: Call delete_syftbox()...")
    manager_do.delete_syftbox(broadcast_delete_events=False)
    manager_ds.delete_syftbox(broadcast_delete_events=False)
    print("  Done.")

    print("\nStep 6: Show state AFTER delete...")
    _show_local_folder(FIXED_DO_PATH, "DO (after delete)")
    query_syftbox_on_gdrive(token_path_do, f"DO ({EMAIL_DO}) - AFTER delete")

    print("\n\n--- KEY FINDING ---")
    private_datasets = FIXED_DO_PATH / "private" / "syft_datasets"
    public_datasets = FIXED_DO_PATH / EMAIL_DO / "public" / "syft_datasets"
    if private_datasets.exists() or public_datasets.exists():
        print(
            "BUG REPRODUCED: Local dataset folders still exist after delete_syftbox()!"
        )
        if private_datasets.exists():
            print(f"  - {private_datasets}")
        if public_datasets.exists():
            print(f"  - {public_datasets}")
    else:
        print("Local folders were cleaned up (bug not reproduced with this config)")


def step_query():
    """Just query GDrive for leftover SyftBox files."""
    query_syftbox_on_gdrive(token_path_do, f"DO ({EMAIL_DO})")
    query_syftbox_on_gdrive(token_path_ds, f"DS ({EMAIL_DS})")
    _show_local_folder(FIXED_DO_PATH, "DO")
    _show_local_folder(FIXED_DS_PATH, "DS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["setup", "query"],
        required=True,
        help="'setup' = create data + delete syftboxes, 'query' = check state",
    )
    args = parser.parse_args()

    if args.step == "setup":
        step_setup()
    elif args.step == "query":
        step_query()
