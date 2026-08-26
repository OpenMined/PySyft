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

from syft_client.sync.environments.environment import Environment
from syft_client.sync.login import _init_client_login, _resolve_login_params
from syft_client.sync.login_utils import handle_potential_version_mismatches_on_login
from syft_client.sync.utils.syftbox_utils import check_env

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
) -> SyftRDSClient:
    """Shared RDS login.

    Mirrors the pre-split ``syft_client`` login flow: detect the environment,
    resolve login params
    """
    env = check_env()
    email, token_path = _resolve_login_params(email, token_path)
    handle_potential_version_mismatches_on_login(email, token_path)

    if env == Environment.COLAB:
        config = SyftRDSClientConfig.for_colab(
            email=email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )
    else:
        config = SyftRDSClientConfig.for_jupyter(
            email=email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            token_path=Path(token_path) if token_path is not None else None,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )

    client = SyftRDSClient.from_config(config)
    _init_client_login(client.sync_engine, sync=sync, load_peers=load_peers)
    return client


def login_do(
    email: str | None = None,
    token_path: str | Path | None = None,
    *,
    sync: bool = True,
    load_peers: bool = True,
    skip_peer_on_patch_version_diff: bool | None = None,
) -> SyftRDSClient:
    """Log in as a Data Owner and return a self-contained RDS client."""
    return _login(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
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
) -> SyftRDSClient:
    """Log in as a Data Scientist and return a self-contained RDS client."""
    return _login(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        has_do_role=False,
        has_ds_role=True,
    )
