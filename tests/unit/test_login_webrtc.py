"""Login with webrtc: the signaling server comes from the settings, and no
Google login or Drive check runs."""

from pathlib import Path

import pytest

from syft.sync import login
from syft.sync.connections.config import ConnectionRouterConfig
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.sync.collection_spec import CollectionSyncSpec
from syft.sync.syftbox_manager import SyftboxManagerConfig

EMAIL = "do@example.com"
URL = "wss://signaling.example.com"
KEY = "test-key"


def test_webrtc_login_reads_signaling_settings(monkeypatch):
    def read_drive(*args):
        raise AssertionError("A WebRTC login must not read Drive")

    monkeypatch.setattr(
        login, "handle_potential_version_mismatches_on_login", read_drive
    )
    monkeypatch.setattr(login.settings, "signaling_url", URL)
    monkeypatch.setattr(login.settings, "signaling_access_key", KEY)

    target = login._resolve_login_target(EMAIL, None, webrtc=True)

    assert target.webrtc == WebRTCConfig(signaling_url=URL, access_key=KEY)
    assert target.repair_pending is False


def test_webrtc_login_without_signaling_url_raises(monkeypatch):
    monkeypatch.setattr(login.settings, "signaling_url", None)
    monkeypatch.setattr(login.settings, "signaling_access_key", KEY)

    with pytest.raises(ValueError, match="SYFT_SIGNALING_URL"):
        login._resolve_login_target(EMAIL, None, webrtc=True)


def test_jupyter_config_with_webrtc_uses_it_for_every_connection():
    webrtc = WebRTCConfig(signaling_url=URL, access_key=KEY)

    config = SyftboxManagerConfig.for_jupyter(EMAIL, has_do_role=True, webrtc=webrtc)

    expected = ConnectionRouterConfig.one_connection(webrtc)
    assert config.connection_router_config == expected


def test_webrtc_config_with_collection_specs_raises():
    spec = CollectionSyncSpec.public("syft_datasets", Path("datasets"))
    webrtc = WebRTCConfig(signaling_url=URL, access_key=KEY)

    with pytest.raises(ValueError, match="Datasets do not go over WebRTC"):
        SyftboxManagerConfig.for_jupyter(
            EMAIL, has_ds_role=True, collection_specs=[spec], webrtc=webrtc
        )
