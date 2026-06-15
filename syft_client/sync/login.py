from pathlib import Path

from syft_client.sync.utils.syftbox_utils import check_env
from syft_client.sync.environments.environment import Environment
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.utils.print_utils import (
    print_client_connected,
    print_client_connecting,
)
from syft_client.sync.utils.syftbox_utils import get_email_colab
from syft_client.sync.config.config import settings
from syft_client.sync.login_utils import handle_potential_version_mismatches_on_login


def _verify_token_matches_email(client: SyftboxManager) -> None:
    """Raise if the email login was called with doesn't match the token's account."""
    actual = client._connection_router.get_authenticated_email()
    if actual.lower() != client.email.lower():
        raise ValueError(
            f"Token/email mismatch: login was called with email={client.email!r} "
            f"but the provided token authenticates as {actual!r}. "
            f"Check that the token file matches the email."
        )


def _init_client_login(
    client: SyftboxManager, sync: bool, load_peers: bool
) -> SyftboxManager:
    """Common post-creation initialization: write version, sync, load peers."""
    _verify_token_matches_email(client)
    print_client_connecting(client.email)
    client.write_local_version()

    if sync:
        client.sync()
    if load_peers:
        client.load_peers()
    print_client_connected(client)
    return client


def _resolve_login_params(
    email: str | None, token_path: str | Path | None
) -> tuple[str, str | Path | None]:
    """Resolve email and token_path based on environment."""
    env = check_env()

    if env == Environment.COLAB:
        if email is None:
            email = get_email_colab()
        if email is None:
            raise ValueError("Email is required for Colab login")
    elif env == Environment.JUPYTER:
        token_path = token_path or settings.token_path
        if not token_path:
            raise NotImplementedError(
                "Jupyter login is only supported with a token path"
            )
        if email is None:
            raise ValueError("Email is required for Jupyter login")
    else:
        raise ValueError(f"Environment {env} not supported")

    return email, token_path


def login(
    email: str | None = None,
    sync: bool = True,
    load_peers: bool = True,
    token_path: str | Path | None = None,
    skip_peer_on_patch_version_diff: bool
    | None = None,  # None: value is determined by the role
):
    return login_ds(
        email,
        sync,
        load_peers,
        token_path,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
    )


def login_ds(
    email: str | None = None,
    sync: bool = True,
    load_peers: bool = True,
    token_path: str | Path | None = None,
    skip_peer_on_patch_version_diff: bool
    | None = None,  # None: value is determined by the role
):
    env = check_env()
    email, token_path = _resolve_login_params(email, token_path)

    handle_potential_version_mismatches_on_login(email, token_path)

    if env == Environment.COLAB:
        client = SyftboxManager.for_colab(
            email=email,
            has_ds_role=True,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )
    else:
        client = SyftboxManager.for_jupyter(
            email=email,
            has_ds_role=True,
            token_path=token_path,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )

    return _init_client_login(client, sync, load_peers)


def login_do(
    email: str | None = None,
    sync: bool = True,
    load_peers: bool = True,
    token_path: str | Path | None = None,
    skip_peer_on_patch_version_diff: bool
    | None = None,  # None: value is determined by the role
):
    env = check_env()
    email, token_path = _resolve_login_params(email, token_path)

    handle_potential_version_mismatches_on_login(email, token_path)

    if env == Environment.COLAB:
        client = SyftboxManager.for_colab(
            email=email,
            has_do_role=True,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )
    else:
        client = SyftboxManager.for_jupyter(
            email=email,
            has_do_role=True,
            token_path=token_path,
            skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        )

    return _init_client_login(client, sync, load_peers)
