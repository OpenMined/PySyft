"""Login helpers for enclave-flow participants."""

from pathlib import Path

from syft_client.sync.environments.environment import Environment
from syft_client.sync.login import _init_client_login, _resolve_login_params
from syft_client.sync.login_utils import handle_potential_version_mismatches_on_login
from syft_client.sync.syftbox_manager import SyftboxManagerConfig
from syft_client.sync.utils.syftbox_utils import check_env

from syft_enclaves.client import SyftEnclaveClient


def _login(
    *,
    email: str | None,
    token_path: str | Path | None,
    sync: bool,
    load_peers: bool,
    skip_peer_on_patch_version_diff: bool | None,
    has_do_role: bool,
    has_ds_role: bool,
    encryption: bool = False,
    encryption_keys: dict | None = None,
) -> SyftEnclaveClient:
    """Shared login for enclave-flow participants.

    Mirrors ``syft_client.login_do``: detect the environment, resolve params,
    run the login-time version-mismatch check, then build the enclave client
    for Colab or Jupyter — always wrapping the job_client with EnclaveJobClient.

    When ``encryption`` is set, keys are persisted to ``~/.syftbox/crypto_keys.json``
    (a stable identity across sessions).
    """
    env = check_env()
    email, token_path = _resolve_login_params(email, token_path)
    handle_potential_version_mismatches_on_login(email, token_path)

    if env == Environment.COLAB:
        config = SyftboxManagerConfig.for_colab(
            email=email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            encryption=encryption,
            encryption_keys=encryption_keys,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )
    else:
        config = SyftboxManagerConfig.for_jupyter(
            email=email,
            has_do_role=has_do_role,
            has_ds_role=has_ds_role,
            token_path=Path(token_path) if token_path is not None else None,
            encryption=encryption,
            encryption_keys=encryption_keys,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )

    # Encryption keys (when enabled) are resolved inside from_config via the
    # config's peer_manager_config.
    client = SyftEnclaveClient.from_config(config)

    # Reuses syft-client's login init: verifies the token authenticates as
    # `email`, writes the local version, then syncs / loads peers.
    _init_client_login(client._manager, sync=sync, load_peers=load_peers)
    return client


def login_do(
    email: str | None = None,
    token_path: str | Path | None = None,
    sync: bool = True,
    load_peers: bool = True,
    skip_peer_on_patch_version_diff: bool | None = None,
    encryption: bool = False,
    encryption_keys: dict | None = None,
) -> SyftEnclaveClient:
    """Log in a data owner for an enclave computation.

    Set ``encryption=True`` to encrypt all drive communication.
    """
    return _login(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        has_do_role=True,
        has_ds_role=True,
        encryption=encryption,
        encryption_keys=encryption_keys,
    )


def login_ds(
    email: str | None = None,
    token_path: str | Path | None = None,
    sync: bool = True,
    load_peers: bool = True,
    skip_peer_on_patch_version_diff: bool | None = None,
    encryption: bool = False,
    encryption_keys: dict | None = None,
) -> SyftEnclaveClient:
    """Log in a data scientist for an enclave computation.

    Set ``encryption=True`` to encrypt all drive communication.
    """
    return _login(
        email=email,
        token_path=token_path,
        sync=sync,
        load_peers=load_peers,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        has_do_role=False,
        has_ds_role=True,
        encryption=encryption,
        encryption_keys=encryption_keys,
    )
