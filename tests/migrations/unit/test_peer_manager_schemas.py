"""PeerManager's live peer-schema maps track loaded peer versions."""

from syft.sync.version.peer_manager import PeerManager
from syft.sync.version.version_info import VersionInfo, VersionInfoV1


def _peer_manager() -> PeerManager:
    # Construct without model_validate side effects: only the private schema
    # map and _update_peer_schemas are exercised here.
    return PeerManager.model_construct()


def _v2_with_schemas() -> VersionInfo:
    return VersionInfo.current()


def _v1() -> VersionInfoV1:
    return VersionInfoV1(
        syft_client_version="0.1.117",
        min_supported_syft_client_version="0.1.93",
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
    )


def test_advertising_peer_appears_in_live_map():
    pm = _peer_manager()
    live = pm.live_peer_schemas("syft-job")
    pm._update_peer_schemas("do@test.org", _v2_with_schemas())
    assert "do@test.org" in live
    assert live["do@test.org"].protocol_name == "syft-job"


def test_pre_v2_peer_is_an_unknown_speaker():
    pm = _peer_manager()
    live = pm.live_peer_schemas("syft-job")
    pm._update_peer_schemas("do@test.org", _v2_with_schemas())
    # A reloaded version file from an old client (upgraded V1: no schemas)
    # must remove the stale entry.
    pm._update_peer_schemas("do@test.org", _v1())
    assert live == {}


def test_cleared_version_removes_peer():
    pm = _peer_manager()
    live = pm.live_peer_schemas("syft")
    pm._update_peer_schemas("do@test.org", _v2_with_schemas())
    pm._update_peer_schemas("do@test.org", None)
    assert live == {}


def test_map_identity_is_stable():
    # live_peer_schemas must always return the same dict object so consumers
    # holding a reference see updates.
    pm = _peer_manager()
    assert pm.live_peer_schemas("syft-job") is pm.live_peer_schemas("syft-job")


def test_late_registered_map_backfills_from_loaded_versions():
    pm = _peer_manager()
    pm._update_peer_schemas("do@test.org", _v2_with_schemas())
    # Registering the protocol AFTER the version loaded must not start empty.
    live = pm.live_peer_schemas("syft-job")
    assert "do@test.org" in live
