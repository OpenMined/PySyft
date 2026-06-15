import time
from syft_client.sync.syftbox_manager import SyftboxManager
import os
from pathlib import Path

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


def remove_syftboxes_from_drive():
    manager_ds, manager_do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=False,
    )
    start = time.time()
    manager_ds.delete_syftbox()
    end = time.time()
    print(f"Time taken to delete syftboxes: {end - start} seconds")
    start = time.time()
    manager_do.delete_syftbox()
    end = time.time()
    print(f"Time taken to delete syftboxes: {end - start} seconds")


def benchmark_gdrive_load_state():
    remove_syftboxes_from_drive()

    manager_ds1, manager_do1 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
            add_peers=True,
            load_peers=False,
        )
    )

    # make some changes
    for i in range(10):
        manager_ds1._send_file_change(f"{EMAIL_DO}/myjob-{i}.job", "Hello, world!")

    time.sleep(3)
    manager_do1.sync()

    # test loading the peers and loading the inbox
    print("initializing second manager")
    manager_ds2, manager_do2 = (
        SyftboxManager._pair_with_google_drive_testing_connection(
            do_email=EMAIL_DO,
            ds_email=EMAIL_DS,
            do_token_path=token_path_do,
            ds_token_path=token_path_ds,
            add_peers=False,
            load_peers=True,
        )
    )
    print(len(manager_do2.datasite_owner_syncer.event_cache.get_cached_events()))

    print("initial sync")
    start = time.time()

    # sync so we have something in the syftbox and do outbox
    manager_do2.sync()

    end = time.time()

    assert len(manager_do2.datasite_owner_syncer.event_cache.get_cached_events()) == 10

    print(f"Time taken to load state: {round(end - start, 2)} seconds")


if __name__ == "__main__":
    benchmark_gdrive_load_state()
