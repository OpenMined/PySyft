"""Login with webrtc: the Firebase project comes from the settings, and no
Google login or Drive check runs."""

from pathlib import Path

import pytest

from syft.sync import login
from syft.sync.connections.config import ConnectionRouterConfig
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.sync.collection_spec import CollectionSyncSpec
from syft.sync.syftbox_manager import SyftboxManagerConfig

EMAIL = "do@example.com"
PROJECT_ID = "syft-test"
API_KEY = "test-api-key"


def test_webrtc_login_reads_firebase_settings(monkeypatch):
    def read_drive(*args):
        raise AssertionError("A WebRTC login must not read Drive")

    monkeypatch.setattr(
        login, "handle_potential_version_mismatches_on_login", read_drive
    )
    monkeypatch.setattr(login.settings, "firebase_project_id", PROJECT_ID)
    monkeypatch.setattr(login.settings, "firebase_api_key", API_KEY)

    target = login._resolve_login_target(EMAIL, None, webrtc=True)

    assert target.webrtc == WebRTCConfig(
        firebase_project_id=PROJECT_ID, firebase_api_key=API_KEY
    )
    assert target.repair_pending is False


def test_webrtc_login_without_firebase_project_raises(monkeypatch):
    monkeypatch.setattr(login.settings, "firebase_project_id", None)
    monkeypatch.setattr(login.settings, "firebase_api_key", API_KEY)

    with pytest.raises(ValueError, match="SYFT_FIREBASE_PROJECT_ID"):
        login._resolve_login_target(EMAIL, None, webrtc=True)


def test_webrtc_login_with_token_path_raises(monkeypatch):
    monkeypatch.setattr(login.settings, "firebase_project_id", PROJECT_ID)
    monkeypatch.setattr(login.settings, "firebase_api_key", API_KEY)

    with pytest.raises(ValueError, match="token_path"):
        login._resolve_login_target(EMAIL, "token.json", webrtc=True)


def test_jupyter_config_with_webrtc_uses_it_for_every_connection():
    webrtc = WebRTCConfig(firebase_project_id=PROJECT_ID, firebase_api_key=API_KEY)

    config = SyftboxManagerConfig.for_jupyter(EMAIL, has_do_role=True, webrtc=webrtc)

    expected = ConnectionRouterConfig.one_connection(webrtc)
    assert config.connection_router_config == expected


def test_webrtc_config_with_collection_specs_raises():
    spec = CollectionSyncSpec.public("syft_datasets", Path("datasets"))
    webrtc = WebRTCConfig(firebase_project_id=PROJECT_ID, firebase_api_key=API_KEY)

    with pytest.raises(ValueError, match="Datasets do not go over WebRTC"):
        SyftboxManagerConfig.for_jupyter(
            EMAIL, has_ds_role=True, collection_specs=[spec], webrtc=webrtc
        )
