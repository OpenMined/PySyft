"""A peer request names the account that owns the request folder.

Any Google account can make a folder whose name names another account, and
share it with a victim. The name alone must not create a request.
"""

import pytest

from syft.sync.connections.drive.gdrive_transport import (
    GDRIVE_P2P_FOLDER_DATASITE_PREFIX,
    PEERS_META_KEY,
    GdriveP2PFolder,
)
from syft.sync.connections.drive.mock_drive_service import (
    GOOGLE_FOLDER_MIME_TYPE,
    MockDriveFile,
    MockPermission,
)
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.version.peer_manager import PeerState
from syft.version import SYFT_VERSION

DO_EMAIL = "do@test.com"
DS_EMAIL = "ds@test.com"
ATTACKER = "eve@attacker.com"
CLAIMED = "alice@partner.org"


def make_do() -> SyftboxManager:
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        email1=DO_EMAIL,
        email2=DS_EMAIL,
        add_peers=False,
    )
    return do_manager


def p2p_name(datasite_email: str, peer_email: str) -> str:
    return (
        f"{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#{SYFT_VERSION}"
        f"#{datasite_email}#inbox#{peer_email}"
    )


def share_request_folder(
    victim: SyftboxManager, owner_email: str | None, claimed_email: str
) -> None:
    """Put a request folder for `victim` in Drive and share it with `victim`."""
    backing_store = victim._connection_router.connections[
        0
    ].drive_service._backing_store
    owners = [{"emailAddress": owner_email}] if owner_email else []
    folder = MockDriveFile(
        name=p2p_name(victim.email, claimed_email),
        mimeType=GOOGLE_FOLDER_MIME_TYPE,
        owners=owners,
    )
    backing_store.add_file(folder)
    backing_store.add_permission(
        folder.id,
        MockPermission(type="user", role="writer", emailAddress=victim.email),
    )


def request_emails(manager: SyftboxManager) -> list[str]:
    return [p.email for p in manager.peer_manager.requested_by_peer_peers]


def test_folder_owned_by_other_account_gives_no_request():
    do_manager = make_do()
    share_request_folder(do_manager, owner_email=ATTACKER, claimed_email=CLAIMED)

    do_manager.load_peers()

    assert CLAIMED not in request_emails(do_manager)


def test_folder_owned_by_other_account_does_not_accept_invited_peer():
    do_manager = make_do()
    do_manager.add_peer(CLAIMED, sync=False)
    assert do_manager.peer_manager.get_cached_peer(CLAIMED).state == (
        PeerState.REQUESTED_BY_ME
    )
    share_request_folder(do_manager, owner_email=ATTACKER, claimed_email=CLAIMED)

    do_manager.load_peers()

    assert do_manager.peer_manager.get_cached_peer(CLAIMED).state == (
        PeerState.REQUESTED_BY_ME
    )


def test_folder_with_no_owner_gives_no_request():
    do_manager = make_do()
    share_request_folder(do_manager, owner_email=None, claimed_email=CLAIMED)

    do_manager.load_peers()

    assert CLAIMED not in request_emails(do_manager)


def test_request_listed_when_name_case_differs_from_owner():
    do_manager = make_do()
    share_request_folder(do_manager, owner_email=CLAIMED, claimed_email=CLAIMED.upper())

    do_manager.load_peers()

    assert request_emails(do_manager) == [CLAIMED.upper()]


@pytest.mark.parametrize(
    "peer_email",
    [
        "alice@partner.org' or name contains '",
        "alice@partner.org/../x",
        "alice@partner.org\\x",
        'alice"@partner.org',
        "alice @partner.org",
        "alice\x00@partner.org",
        "alice@@partner.org",
        "alice@partner",
        "..",
        PEERS_META_KEY,
    ],
)
def test_from_name_rejects_bad_peer_email(peer_email):
    with pytest.raises(ValueError):
        GdriveP2PFolder.from_name(p2p_name(DO_EMAIL, peer_email))


def test_from_name_rejects_bad_datasite_email():
    with pytest.raises(ValueError):
        GdriveP2PFolder.from_name(p2p_name("do'@test.com", CLAIMED))


@pytest.mark.parametrize(
    "peer_email", ["Alice.Smith+syft@sub.partner.co.uk", "a_b-c@x.io"]
)
def test_from_name_accepts_email(peer_email):
    assert GdriveP2PFolder.from_name(p2p_name(DO_EMAIL, peer_email)).peer_email == (
        peer_email
    )
