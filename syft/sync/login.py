from dataclasses import dataclass
from pathlib import Path

from syft.sync.config.config import settings
from syft.sync.connections.webrtc.firebase.auth import (
    firebase_session_path,
    session_is_refused,
    sign_in_with_emailed_link,
)
from syft.sync.connections.webrtc.firebase.project import FirebaseProject
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.environments.environment import Environment
from syft.sync.login_utils import handle_potential_version_mismatches_on_login
from syft.sync.peers.peer_store import datasite_crypto_keys_path, import_key_file
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from syft.sync.utils.print_utils import (
    print_client_connected,
    print_client_connecting,
)
from syft.sync.utils.syftbox_utils import check_env, get_email_colab


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
    client.peer_manager.write_own_peer_info()

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
                "Running outside Colab: pass token_path=... to login_do()/login_ds(). "
                "See docs/auth.md to create a token (about 15 minutes)."
            )
        if email is None:
            raise ValueError("Email is required for Jupyter login")
    else:
        raise ValueError(f"Environment {env} not supported")

    return email, token_path


@dataclass(frozen=True)
class LoginTarget:
    email: str
    # None on Colab, and in WebRTC mode.
    token_path: str | Path | None
    # True when a version mismatch kept the data, and the state needs a repair.
    repair_pending: bool
    # None: the client syncs over Google Drive.
    webrtc: WebRTCConfig | None


def _resolve_login_target(
    email: str | None, token_path: str | Path | None, webrtc: bool | WebRTCConfig
) -> LoginTarget:
    """Resolve what a login connects to.

    Raises:
        ValueError: ``webrtc`` is set and ``email`` is None or ``token_path``
            is set, or ``webrtc`` is True and a Firebase setting is missing.
    """
    if webrtc is False:
        email, token_path = _resolve_login_params(email, token_path)
        repair_pending = handle_potential_version_mismatches_on_login(email, token_path)
        return LoginTarget(email, token_path, repair_pending, None)
    if email is None:
        raise ValueError("Pass your email to log in with webrtc.")
    if token_path is not None:
        raise ValueError(f"A WebRTC login takes no token_path, got {token_path!r}")
    webrtc_config = _webrtc_config_from_settings() if webrtc is True else webrtc
    return LoginTarget(email, None, False, webrtc_config)


def _webrtc_config_from_settings() -> WebRTCConfig:
    """Return the Firebase project in the settings.

    Raises:
        ValueError: A Firebase setting is missing.
    """
    if settings.firebase_project_id is None:
        raise ValueError(
            "Set SYFT_FIREBASE_PROJECT_ID to the Firebase project of your org, "
            "then log in again."
        )
    if settings.firebase_api_key is None:
        raise ValueError(
            "Set SYFT_FIREBASE_API_KEY to the web API key of the Firebase project "
            "of your org, then log in again."
        )
    return WebRTCConfig(
        firebase_project_id=settings.firebase_project_id,
        firebase_api_key=settings.firebase_api_key,
    )


def _maybe_sign_in_to_firebase(target: LoginTarget, syftbox_folder: Path) -> None:
    """Sign in to the Firebase project of a WebRTC login with an emailed link,
    if this machine has no session for it, or Firebase refuses the saved one."""
    if target.webrtc is None:
        return
    project = FirebaseProject.from_config(target.webrtc)
    session_path = firebase_session_path(syftbox_folder, project.project_id)
    if session_path.exists() and session_is_refused(project, session_path):
        session_path.unlink()
    if not session_path.exists():
        sign_in_with_emailed_link(
            project, target.email, session_path, lambda: _ask_sign_in_link(target.email)
        )


def _ask_sign_in_link(email: str) -> str:
    return input(f"We sent a sign-in link to {email}. Paste the link here: ")


def _import_key_file(
    key_file: str | Path | None, syftbox_folder: Path, email: str
) -> None:
    """Copy ``key_file``, the key file of an earlier session, into the datasite of ``email``.

    Call it before the client is built, so that the client loads these keys.
    """
    if key_file is not None:
        import_key_file(
            key_file, datasite_crypto_keys_path(syftbox_folder, email), email
        )


def login(
    email: str | None = None,
    sync: bool = True,
    load_peers: bool = True,
    token_path: str | Path | None = None,
    skip_peer_on_patch_version_diff: bool
    | None = None,  # None: value is determined by the role
    encryption: bool | None = None,  # True: required; None: the datasite's policy
    key_file: str | Path | None = None,  # crypto_keys.json of an earlier session
    webrtc: bool | WebRTCConfig = False,  # True: the Firebase project in the settings
):
    return login_ds(
        email,
        sync,
        load_peers,
        token_path,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        encryption=encryption,
        key_file=key_file,
        webrtc=webrtc,
    )


def login_ds(
    email: str | None = None,
    sync: bool = True,
    load_peers: bool = True,
    token_path: str | Path | None = None,
    skip_peer_on_patch_version_diff: bool
    | None = None,  # None: value is determined by the role
    encryption: bool | None = None,  # True: required; None: the datasite's policy
    key_file: str | Path | None = None,  # crypto_keys.json of an earlier session
    webrtc: bool | WebRTCConfig = False,  # True: the Firebase project in the settings
):
    target = _resolve_login_target(email, token_path, webrtc)
    config = _manager_config(
        target,
        has_ds_role=True,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        encryption=encryption,
    )
    _import_key_file(key_file, config.syftbox_folder, target.email)
    _maybe_sign_in_to_firebase(target, config.syftbox_folder)
    client = SyftboxManager.from_config(config)

    return _init_client_login(client, sync, load_peers, target.repair_pending)


def login_do(
    email: str | None = None,
    sync: bool = True,
    load_peers: bool = True,
    token_path: str | Path | None = None,
    skip_peer_on_patch_version_diff: bool
    | None = None,  # None: value is determined by the role
    encryption: bool | None = None,  # True: required; None: the datasite's policy
    key_file: str | Path | None = None,  # crypto_keys.json of an earlier session
    webrtc: bool | WebRTCConfig = False,  # True: the Firebase project in the settings
):
    target = _resolve_login_target(email, token_path, webrtc)
    config = _manager_config(
        target,
        has_do_role=True,
        skip_peer_on_patch_version_diff=skip_peer_on_patch_version_diff,
        encryption=encryption,
    )
    _import_key_file(key_file, config.syftbox_folder, target.email)
    _maybe_sign_in_to_firebase(target, config.syftbox_folder)
    client = SyftboxManager.from_config(config)

    return _init_client_login(client, sync, load_peers, target.repair_pending)


def _manager_config(target: LoginTarget, **config_kwargs) -> SyftboxManagerConfig:
    """The config of a manager that logs in to ``target``, on Colab or in
    Jupyter."""
    if check_env() == Environment.COLAB:
        return SyftboxManagerConfig.for_colab(
            email=target.email, webrtc=target.webrtc, **config_kwargs
        )
    token_path = Path(target.token_path) if target.token_path is not None else None
    return SyftboxManagerConfig.for_jupyter(
        email=target.email, token_path=token_path, webrtc=target.webrtc, **config_kwargs
    )
