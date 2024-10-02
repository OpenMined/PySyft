# syft absolute
from syft.abstract_server import ServerType
from syft.server.credentials import SyftSigningKey
from syft.service.network.network_service import NetworkStash
from syft.service.network.server_peer import ServerPeer
from syft.service.network.server_peer import ServerPeerUpdate
from syft.types.uid import UID


def test_add_route() -> None:
    uid = UID()
    peer = ServerPeer(
        id=uid,
        name="test",
        verify_key=SyftSigningKey.generate().verify_key,
        server_type=ServerType.DATASITE,
        admin_email="info@openmined.org",
    )
    network_stash = NetworkStash.random()

    network_stash.set(
        credentials=network_stash.db.root_verify_key,
        obj=peer,
    ).unwrap()
    peer_update = ServerPeerUpdate(id=uid, name="new name")
    peer = network_stash.update(
        credentials=network_stash.db.root_verify_key,
        obj=peer_update,
    ).unwrap()

    assert peer.name == "new name"
