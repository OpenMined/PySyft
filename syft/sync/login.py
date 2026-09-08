from pathlib import Path

from syft.sync.utils.syftbox_utils import check_env
from syft.sync.environments.environment import Environment
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.utils.print_utils import (
    print_client_connected,
    print_client_connecting,
)
from syft.sync.utils.syftbox_utils import get_email_colab
from syft.sync.config.config import settings
from syft.sync.login_utils import handle_potential_version_mismatches_on_login


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
    client: SyftboxManager,
    sync: bool,
    load_peers: bool,
    repair_pending: bool = False,
) -> SyftboxManager:
    """Common post-creation initialization: write version, sync, load peers.

    `repair_pending` says that a version mismatch kept the data, so the state
    still needs a repair. The repair is lazy: the client adopts a Drive folder
    when it first looks that folder up, and a cache resets when it is read. A
    sync does all of that at once, so this function reports whether one runs.
    """
    _verify_token_matches_email(client)
    print_client_connecting(client.email)
    # Write the version file on both sides. A local-only write leaves the remote
    # file at the version that first created it. Two things then break: the
    # login mismatch check reads that stale file and prompts at every login, and
    # a peer reads it to select a job or dataset protocol version for us.
    client.peer_manager.write_own_version()

    if repair_pending:
        if sync:
            print(
                "Repairing the state now. Drive folders of an earlier client "
                "version are adopted, and caches and checkpoints rebuild "
                "themselves.\n"
            )
        else:
            print(
                "Warning: the state is not repaired yet, because this login "
                "does not sync. Drive folders of an earlier client version are "
                "adopted, and caches and checkpoints rebuild themselves, at the "
                "first sync. Call client.sync() to do it now.\n"
            )

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

    repair_pending = handle_potential_version_mismatches_on_login(email, token_path)

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

    return _init_client_login(client, sync, load_peers, repair_pending)


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

    repair_pending = handle_potential_version_mismatches_on_login(email, token_path)

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

    return _init_client_login(client, sync, load_peers, repair_pending)
