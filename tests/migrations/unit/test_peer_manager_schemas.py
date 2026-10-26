"""PeerStore.protocol_schema reads the schemas a peer advertises in its peer info."""

from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.peers.peer import Peer
from syft.sync.peers.peer_store import PeerStore
from syft.sync.version.peer_info import PeerInfo
from syft.sync.version.version_info import VersionInfo, VersionInfoV1

PEER = "do@test.org"


def _store_with_peer(version_info) -> PeerStore:
    store = PeerStore(email="ds@test.org")
    info = None
    if version_info is not None:
        info = PeerInfo.model_construct(
            version_info=version_info, encryption_policy=EncryptionPolicy.OPTIONAL
        )
    store.add_peer(Peer(email=PEER, info=info))
    return store


def _v1() -> VersionInfoV1:
    return VersionInfoV1(
        syft_client_version="0.1.117",
        min_supported_syft_client_version="0.1.93",
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
    )


def test_advertising_peer_has_schema():
    store = _store_with_peer(VersionInfo.current())
    assert store.protocol_schema(PEER, "syft-job").protocol_name == "syft-job"


def test_pre_v2_peer_is_an_unknown_speaker():
    # A version file from an old client (V1) advertises no schemas.
    assert _store_with_peer(_v1()).protocol_schema(PEER, "syft-job") is None


def test_peer_without_info_has_no_schema():
    assert _store_with_peer(None).protocol_schema(PEER, "syft") is None


def test_peer_not_in_store_has_no_schema():
    assert PeerStore(email="ds@test.org").protocol_schema(PEER, "syft") is None


def test_schema_follows_reloaded_info():
    store = _store_with_peer(VersionInfo.current())
    store.get_cached_peer(PEER).info = None
    assert store.protocol_schema(PEER, "syft-job") is None
