"""Entry points for the Remote Data Science product.

    from syft_rds import login_do
    rds_client = login_do(email, token_path)
    rds_client.datasets
    rds_client.jobs

These wrap the syft-client sync-engine login and return a composed
``SyftRDSClient``. Note the argument order: ``token_path`` is the 2nd positional
here (matching the product interface), whereas the underlying sync-engine login
takes it as a keyword.
"""

from __future__ import annotations

from pathlib import Path

from syft_client.sync.login import login_do as _sync_login_do
from syft_client.sync.login import login_ds as _sync_login_ds

from syft_rds.client import SyftRDSClient


def login_do(
    email: str | None = None,
    token_path: str | Path | None = None,
    *,
    sync: bool = True,
    load_peers: bool = True,
    skip_peer_on_patch_version_diff: bool | None = None,
) -> SyftRDSClient:
    """Log in as a Data Owner and return a composed RDS client."""
    sync_engine = _sync_login_do(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
    )
    return SyftRDSClient(sync_engine)


def login_ds(
    email: str | None = None,
    token_path: str | Path | None = None,
    *,
    sync: bool = True,
    load_peers: bool = True,
    skip_peer_on_patch_version_diff: bool | None = None,
) -> SyftRDSClient:
    """Log in as a Data Scientist and return a composed RDS client."""
    sync_engine = _sync_login_ds(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
    )
    return SyftRDSClient(sync_engine)
