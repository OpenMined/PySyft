"""P2P folder lookup accepts any client version in the folder name.

A P2P folder name is a rendezvous string that both peers compute from their own
client version, so neither side may rename it (see the adopt path for private
folders). Lookup therefore has to tolerate the version instead.

Reuse matters in both directions. A folder this client owns must be reused after
an upgrade, because a peer that still filters by name would not find a new one.
A folder the peer owns must be found whatever version the peer wrote into it.
"""

from unittest.mock import Mock

from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection

ME = "alice@example.com"
PEER = "bob@example.com"


def _name(version: str, datasite: str, folder_type: str, peer: str) -> str:
    return f"syft_datasite#{version}#{datasite}#{folder_type}#{peer}"


def _conn(found):
    conn = GDriveConnection(email=ME, verbose=False)
    conn.drive_service = Mock()
    conn._find_folders = Mock(return_value=found)
    return conn


def _lookup(conn):
    return conn._find_p2p_folder_id(
        datasite_email=PEER, folder_type="inbox", peer_email=ME, owner_email=ME
    )


def test_a_folder_of_another_minor_version_is_found():
    # The old filter dropped this folder, so the client created a second one and
    # the peer kept writing into the first. 0.2.0 differs in the minor from the
    # current client version, which is what the filter used to reject.
    old = _name("0.2.0", PEER, "inbox", ME)
    assert _lookup(_conn([("old", old)])) == "old"


def test_a_folder_of_an_older_major_version_is_found():
    old = _name("0.0.9", PEER, "inbox", ME)
    assert _lookup(_conn([("old", old)])) == "old"


def test_the_highest_version_wins_when_several_exist():
    folders = [
        ("v1", _name("0.1.117", PEER, "inbox", ME)),
        ("v2", _name("0.2.0", PEER, "inbox", ME)),
        ("v0", _name("0.0.9", PEER, "inbox", ME)),
    ]
    assert _lookup(_conn(folders)) == "v2"


def test_versions_order_by_number_not_by_string():
    folders = [
        ("nine", _name("0.1.9", PEER, "inbox", ME)),
        ("ten", _name("0.1.10", PEER, "inbox", ME)),
    ]
    assert _lookup(_conn(folders)) == "ten"


def test_several_folders_no_longer_raise():
    folders = [
        ("a", _name("0.1.117", PEER, "inbox", ME)),
        ("b", _name("0.1.118", PEER, "inbox", ME)),
    ]
    assert _lookup(_conn(folders)) is not None


def test_a_folder_of_another_peer_is_ignored():
    other = _name("0.1.117", PEER, "inbox", "carol@example.com")
    assert _lookup(_conn([("other", other)])) is None


def test_a_folder_of_another_type_is_ignored():
    outbox = _name("0.1.117", PEER, "outbox", ME)
    assert _lookup(_conn([("outbox", outbox)])) is None


def test_no_folder_returns_none():
    assert _lookup(_conn([])) is None


def test_a_name_that_does_not_parse_is_ignored():
    assert _lookup(_conn([("junk", "not_a_p2p_folder")])) is None
