"""
Integration tests without unit test coverage.

These tests are unique to integration testing and verify behavior that
cannot be adequately tested with mocked connections.
"""

from syft.sync.peers.exceptions import SyftPeerNotReadyError
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.utils.waiting import poll_until
import os
import time
from pathlib import Path
import pytest


SYFT_DIR = Path(__file__).parent.parent.parent.parent
# These are in gitignore, create yourself
CREDENTIALS_DIR = SYFT_DIR / "credentials"

# koen gmail
FILE_DO = os.environ.get("ai_audit_credentials_fname_do", "token_do.json")
EMAIL_DO = os.environ["AI_AUDIT_EMAIL_DO"]

# koen openmined mail
FILE_DS = os.environ.get("ai_audit_credentials_fname_ds", "token_ds.json")
EMAIL_DS = os.environ["AI_AUDIT_EMAIL_DS"]


token_path_do = CREDENTIALS_DIR / FILE_DO
token_path_ds = CREDENTIALS_DIR / FILE_DS

# Drive lists a new file some time after the write.
POLL_TIMEOUT = 60
POLL_INTERVAL = 1


def _poll(check, what: str):
    """Call ``check`` until it returns a value that is not None."""
    result = poll_until(check, time.monotonic() + POLL_TIMEOUT, POLL_INTERVAL)
    assert result is not None, f"Timed out after {POLL_TIMEOUT}s waiting for {what}"
    return result


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_peer_request_blocks_sync_until_approved():
    """
    Integration test: Files don't sync until peer request is approved.

    Workflow:
    1. DS adds DO as peer (creates peer request)
    2. DS submits a job - the send raises (DO policy not known)
    3. DO syncs - nothing should sync (peer not approved)
    4. DO approves peer request
    5. DS syncs and submits again, DO syncs - job should now sync
    """
    # Create managers with Google Drive connection, no auto-add peers
    ds_manager, do_manager = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=False,  # Don't auto-add - we'll test the request flow
    )

    # Grant DS write permissions on DO's datasite
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    # Step 1: DS makes peer request by adding DO
    ds_manager.add_peer(do_manager.email)

    # Verify: DO sees this as a pending request
    def request_seen():
        do_manager.load_peers()
        return do_manager.peer_manager.requested_by_peer_peers or None

    _poll(request_seen, "the peer request")
    assert len(do_manager.peer_manager.requested_by_peer_peers) == 1
    assert len(do_manager.peer_manager.approved_peers) == 0
    assert do_manager.peer_manager.requested_by_peer_peers[0].email == ds_manager.email

    # Step 2: DS submits a simple job
    job_file_path = f"{do_manager.email}/test.job"
    job_content = "print('Hello from DS')"
    with pytest.raises(SyftPeerNotReadyError):
        ds_manager._send_file_change(job_file_path, job_content)

    # Step 3: DO syncs WITHOUT accepting - nothing should sync
    do_manager.sync()

    # Verify: Cache is empty (no messages processed)
    do_cache = do_manager.datasite_owner_syncer.event_cache
    assert len(do_cache.file_hashes) == 0, "Cache should be empty - peer not approved"

    # Step 4: DO approves peer request
    do_manager.approve_peer_request(ds_manager.email)

    # Verify: Peer moved from requests to approved
    assert len(do_manager.peer_manager.requested_by_peer_peers) == 0
    assert len(do_manager.peer_manager.approved_peers) == 1
    assert do_manager.peer_manager.approved_peers[0].email == ds_manager.email

    # Step 5: DS syncs, which reads the DO policy, and submits again. DO syncs
    # again - now it should work
    def resent():
        ds_manager.sync()
        try:
            ds_manager._send_file_change(job_file_path, job_content)
        except SyftPeerNotReadyError:
            return None  # The DS does not see the DO version file yet.
        return True

    _poll(resent, "the DO policy")

    def job_received():
        do_manager.sync()
        return do_cache.file_hashes or None

    # Verify: File synced and in cache
    _poll(job_received, "the job at the DO")

    # Verify: File is tracked in cache with correct path (stored as PosixPath)
    expected_cache_path = Path("test.job")
    assert expected_cache_path in do_cache.file_hashes, (
        f"File {expected_cache_path} should be in cache"
    )

    # Verify: Content is correct
    assert do_cache.file_hashes[expected_cache_path] is not None, (
        "File should have a hash"
    )
