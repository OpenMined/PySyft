import os
from pathlib import Path

from syft_client.sync.syftbox_manager import SyftboxManager


def is_mock_mode() -> bool:
    return os.environ.get("INTEGRATION_TEST_MOCK_MODE", "").lower() == "true"


def create_test_pair(
    do_email: str,
    ds_email: str,
    do_token_path: Path | None = None,
    ds_token_path: Path | None = None,
    add_peers: bool = True,
    load_peers: bool = False,
    use_in_memory_cache: bool = True,
    clear_caches: bool = True,
    check_versions: bool = False,
):
    if is_mock_mode():
        return SyftboxManager.pair_with_mock_drive_service_connection(
            email1=do_email,
            email2=ds_email,
            add_peers=add_peers,
            use_in_memory_cache=use_in_memory_cache,
            check_versions=check_versions,
        )
    return SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=do_email,
        ds_email=ds_email,
        do_token_path=do_token_path,
        ds_token_path=ds_token_path,
        add_peers=add_peers,
        load_peers=load_peers,
        use_in_memory_cache=use_in_memory_cache,
        clear_caches=clear_caches,
        check_versions=check_versions,
    )
