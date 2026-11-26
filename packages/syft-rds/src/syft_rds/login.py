"""Entry points for the Remote Data Science product.

    from syft_rds import login_do
    rds_client = login_do(email, token_path)
    rds_client.datasets
    rds_client.jobs

These build a composed ``SyftRDSClientConfig`` (which owns the sync + job +
dataset sub-configs), construct a self-contained ``SyftRDSClient`` from it
"""

from __future__ import annotations

from pathlib import Path

from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.environments.environment import Environment
from syft.sync.login import (
    _import_key_file,
    _init_client_login,
    _maybe_sign_in_to_firebase,
    _resolve_login_target,
)
from syft.sync.utils.syftbox_utils import check_env
from syft_rds.client import SyftRDSClient
from syft_rds.config import SyftRDSClientConfig


def _login(
    *,
    email: str | None,
    token_path: str | Path | None,
    sync: bool,
    load_peers: bool,
    skip_peer_on_patch_version_diff: bool | None,
    has_do_role: bool,
    has_ds_role: bool,
    encryption: bool | None = None,
    key_file: str | Path | None = None,
    webrtc: bool | WebRTCConfig = False,
) -> SyftRDSClient:
    """Shared RDS login.

    Mirrors the pre-split ``syft`` login flow: detect the environment,
    resolve login params
    """
    target = _resolve_login_target(email, token_path, webrtc)

    if check_env() == Environment.COLAB:
        config = SyftRDSClientConfig.for_colab(
            email=target.email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
            encryption=encryption,
            webrtc=target.webrtc,
        )
    else:
        config = SyftRDSClientConfig.for_jupyter(
            email=target.email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            token_path=Path(target.token_path) if target.token_path else None,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
            encryption=encryption,
            webrtc=target.webrtc,
        )

    _import_key_file(key_file, config.sync.syftbox_folder, target.email)
    _maybe_sign_in_to_firebase(target, config.sync.syftbox_folder)
    client = SyftRDSClient.from_config(config)
    _init_client_login(
        client.sync_engine,
        sync=sync,
        load_peers=load_peers,
        repair_pending=target.repair_pending,
    )
    return client


def login_do(
    email: str | None = None,
    token_path: str | Path | None = None,
    *,
    sync: bool = True,
    load_peers: bool = True,
    skip_peer_on_patch_version_diff: bool | None = None,
    encryption: bool | None = None,
    key_file: str | Path | None = None,
    webrtc: bool | WebRTCConfig = False,
) -> SyftRDSClient:
    """Log in as a DO and return a self-contained RDS client.

    ``webrtc=True`` syncs over WebRTC with no Google login, through the
    Firebase project in the ``SYFT_FIREBASE_PROJECT_ID`` and
    ``SYFT_FIREBASE_API_KEY`` settings. The first login on a machine sends a
    sign-in link to the email.
    """
    return _login(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        encryption=encryption,
        key_file=key_file,
        webrtc=webrtc,
        has_do_role=True,
        has_ds_role=False,
    )


def login_ds(
    email: str | None = None,
    token_path: str | Path | None = None,
    *,
    sync: bool = True,
    load_peers: bool = True,
    skip_peer_on_patch_version_diff: bool | None = None,
    encryption: bool | None = None,
    key_file: str | Path | None = None,
    webrtc: bool | WebRTCConfig = False,
) -> SyftRDSClient:
    """Log in as a DS and return a self-contained RDS client.

    ``webrtc=True`` syncs over WebRTC with no Google login, through the
    Firebase project in the ``SYFT_FIREBASE_PROJECT_ID`` and
    ``SYFT_FIREBASE_API_KEY`` settings. The first login on a machine sends a
    sign-in link to the email.
    """
    return _login(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        encryption=encryption,
        key_file=key_file,
        webrtc=webrtc,
        has_do_role=False,
        has_ds_role=True,
    )
