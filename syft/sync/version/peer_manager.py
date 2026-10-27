"""
PeerManager for managing peers, version information, and compatibility checks.
"""

import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from syft.sync.connections.base_connection import ConnectionConfig
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.peers.peer import Peer, PeerNotReadyError, PeerSetupError, PeerState
from syft.sync.peers.peer_store import (
    PeerStore,
    datasite_crypto_keys_path,
    datasite_settings_path,
    resolve_encryption_policy,
)
from syft.sync.utils.print_utils import (
    print_peer_already_connected,
    print_peer_connection_established,
    print_peer_request_accepting,
    print_peer_request_resending,
    print_peer_request_sending,
    print_peer_request_sent,
)
from syft.sync.version.exceptions import (
    VersionMismatchError,
    VersionUnknownError,
)
from syft.sync.version.peer_info import PeerInfo
from syft.sync.version.version_info import CompatibilityStatus, VersionInfo

logger = logging.getLogger(__name__)


class CompatAction(str, Enum):
    """What the caller is doing — used to phrase warning messages."""

    SYNC = "sync"
    SUBMIT = "submit"
    EXECUTE = "execute"


class PeerCompatibilityResult(BaseModel):
    """Outcome of a per-peer version compatibility check.

    Carries enough context for callers to either warn (`maybe_warn`) or raise
    (`raise_on_skip`) without re-deriving anything from PeerManager state.
    """

    peer_email: str
    status: CompatibilityStatus
    should_skip: bool
    action: Optional[CompatAction] = None
    explanation_skip: Optional[str] = None
    explanation_not_skip: Optional[str] = None
    suppress_version_warnings: bool = False
    local_version: Optional[VersionInfo] = None
    peer_version: Optional[VersionInfo] = None

    # TODO: rename `maybe_warn` -> `maybe_log` and `suppress_version_warnings`
    # -> `suppress_version_logs`. The body now emits via logger.info() rather
    # than warnings.warn(), so the "warn" naming is misleading. Held off on
    # the rename to keep this change minimal; do it in a follow-up that
    # updates the public API and all call sites.
    def maybe_warn(self) -> None:
        """Emit an INFO log, picking explanation_skip vs explanation_not_skip.

        Appends an override hint only when skipping. Respects
        suppress_version_warnings.
        """
        if self.suppress_version_warnings:
            return
        elif self.should_skip:
            if self.explanation_skip:
                logger.info(
                    f"{self.explanation_skip} Use ignore_peer_version=True to override."
                )
        elif self.explanation_not_skip:
            logger.info(self.explanation_not_skip)

    def raise_on_skip(self, operation: str) -> None:
        """Raise the appropriate VersionError if `should_skip` is True."""
        if not self.should_skip:
            return
        elif self.status == CompatibilityStatus.UNKNOWN:
            raise VersionUnknownError(self.peer_email, operation)
        else:
            reason = self.explanation_skip or "version mismatch"
            raise VersionMismatchError(
                self.peer_email, self.local_version, self.peer_version, reason
            )


class PeerManagerConfig(BaseModel):
    """Configuration for PeerManager."""

    model_config = ConfigDict(extra="forbid")

    syftbox_folder: Path
    email: str = ""
    connection_configs: List[ConnectionConfig] = []
    # Proceeds with a peer whose version is unknown or whose client version is
    # incompatible, instead of skipping it.
    force_ignore_peer_version: bool = False
    suppress_version_warnings: bool = False
    n_threads: int = 10
    has_do_role: bool = False
    has_ds_role: bool = False
    # None: the policy of the datasite on disk, or OPTIONAL on a new datasite.
    encryption_policy: Optional[EncryptionPolicy] = None
    skip_peer_on_patch_version_diff: Optional[bool] = (
        None  # None: value is determined by the role (resolves to has_do_role)
    )

    @property
    def crypto_keys_path(self) -> Path:
        """Per-datasite encryption key file for this participant
        (``<syftbox_folder>/<email>/private/crypto_keys.json``)."""
        return datasite_crypto_keys_path(self.syftbox_folder, self.email)

    @property
    def settings_path(self) -> Path:
        """Per-datasite settings file, shared by every process on the datasite."""
        return datasite_settings_path(self.syftbox_folder, self.email)

    @model_validator(mode="after")
    def _default_skip_peer_on_patch(self) -> "PeerManagerConfig":
        if self.skip_peer_on_patch_version_diff is None:
            self.skip_peer_on_patch_version_diff = self.has_do_role
        return self


class PeerManager(BaseModel):
    """Manages version information for self and peers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    syftbox_folder: Path
    connection_router: ConnectionRouter
    peer_store: PeerStore
    force_ignore_peer_version: bool = False
    suppress_version_warnings: bool = False
    n_threads: int = 10
    has_do_role: bool = False
    has_ds_role: bool = False
    skip_peer_on_patch_version_diff: bool = False

    _own_version: Optional[VersionInfo] = PrivateAttr(default=None)
    _executor: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)
    # Set by the first load_peers(), which records Peer.state_at_start.
    _first_load_done: bool = PrivateAttr(default=False)

    # ========== Peer List Properties ==========

    def clear_caches(self) -> None:
        """Clear the caches."""
        self.peer_store.clear_caches()

    @property
    def approved_peers(self) -> List[Peer]:
        """Get all approved peers (DO side)."""
        return self.peer_store.approved_peers

    @property
    def requested_by_peer_peers(self) -> List[Peer]:
        """Get all peers that requested us (DO side)."""
        return self.peer_store.requested_by_peer_peers

    @property
    def requested_by_me_peers(self) -> List[Peer]:
        """Get all peers we requested but haven't reciprocated yet."""
        return self.peer_store.requested_by_me_peers

    @property
    def syncable_peers(self) -> List[Peer]:
        """Get all peers we can sync with (DS side)."""
        return self.peer_store.syncable_peers

    @classmethod
    def from_config(cls, config: PeerManagerConfig, email: str = "") -> "PeerManager":
        """Create a PeerManager from a config.

        ``email`` is a fallback for configs built without one; the config's own
        ``email`` takes precedence so the key path matches the datasite.
        """
        resolved_email = config.email or email
        peer_store = PeerStore.create(
            email=resolved_email,
            encryption_policy=resolve_encryption_policy(
                config.settings_path, config.encryption_policy
            ),
            keys_path=config.crypto_keys_path,
        )
        connection_router = ConnectionRouter.from_configs(
            resolved_email, config.connection_configs
        )
        connection_router.peer_store = peer_store
        return cls(
            syftbox_folder=config.syftbox_folder,
            connection_router=connection_router,
            peer_store=peer_store,
            force_ignore_peer_version=config.force_ignore_peer_version,
            suppress_version_warnings=config.suppress_version_warnings,
            n_threads=config.n_threads,
            has_do_role=config.has_do_role,
            has_ds_role=config.has_ds_role,
            skip_peer_on_patch_version_diff=bool(
                config.skip_peer_on_patch_version_diff
            ),
        )

    def model_post_init(self, __context) -> None:
        """Initialize the thread pool executor."""
        self._executor = ThreadPoolExecutor(max_workers=self.n_threads)

    def get_own_version(self) -> VersionInfo:
        """Get current client's version info."""
        if self._own_version is None:
            self._own_version = VersionInfo.current()
        return self._own_version

    def get_own_peer_info(self) -> PeerInfo:
        """What this client publishes to its peers."""
        return PeerInfo(
            version_info=self.get_own_version(),
            encryption_policy=self.peer_store.encryption_policy,
        )

    def write_own_peer_info(self) -> None:
        """Write the peer info file to own SyftBox folder, and the version to local disk."""
        from syft.sync.version.local_version import write_local_version

        self.connection_router.write_peer_info_file(self.get_own_peer_info())
        write_local_version(self.syftbox_folder)

    def share_peer_info_with_peer(self, peer_email: str) -> None:
        """Share the peer info file with a peer so they can read it."""
        self.connection_router.share_peer_info_file_with_peer(
            peer_email, self.get_own_peer_info()
        )

    def get_cached_peer(self, email: str) -> Optional[Peer]:
        """Get a peer by email, or None if not found."""
        return self.peer_store.get_cached_peer(email)

    def load_peer_info(self, peer_email: str) -> Optional[PeerInfo]:
        """Load the info of a single peer (blocking)."""
        peer_info = self.connection_router.read_peer_info_file(peer_email)
        self._set_peer_info(peer_email, peer_info)
        return peer_info

    def _set_peer_info(self, peer_email: str, peer_info: Optional[PeerInfo]) -> None:
        """Cache a freshly loaded peer info on the peer.

        A read that finds nothing keeps the info known before: a peer that
        deleted its SyftBox still sends messages under its last policy, such
        as the deletes of its files.
        """
        cached_peer = self.get_cached_peer(peer_email)
        if cached_peer and peer_info is not None:
            cached_peer.info = peer_info

    def _load_single_peer_info(self, peer_email: str) -> tuple[str, Optional[PeerInfo]]:
        """Load the info of a single peer. Used by parallel loader.

        Raises:
            TransportUnavailableError: when the backend cannot be reached.
            ValueError: when the document is not one this client reads.
        """
        # Create new connection for thread safety
        connection = self.connection_router.connection_for_peer_file_read(
            create_new=True
        )
        return (peer_email, connection.read_peer_info_file(peer_email))

    def load_peer_infos_parallel(
        self, peer_emails: List[str], force: bool = False
    ) -> Dict[str, Optional[PeerInfo]]:
        """Load the infos of multiple peers in parallel using internal executor."""

        if not peer_emails:
            return {}

        # Submit all tasks and collect results
        results = list(self._executor.map(self._load_single_peer_info, peer_emails))

        for email, peer_info in results:
            self._set_peer_info(email, peer_info)

        return dict(results)

    def get_peer_version(self, peer_email: str) -> Optional[VersionInfo]:
        """Get cached version for peer, None if not loaded or not available."""
        peer = self.get_cached_peer(peer_email)
        return peer.version if peer else None

    def clear_peer_info(self, peer_email: str) -> None:
        """Clear the cached info of a peer (useful for testing)."""
        cached_peer = self.get_cached_peer(peer_email)
        if cached_peer:
            cached_peer.info = None

    def peer_compatibility_status(self, peer_email: str) -> CompatibilityStatus:
        """Get the CompatibilityStatus for a peer (UNKNOWN if version not loaded)."""
        peer_version = self.get_peer_version(peer_email)
        return self.get_own_version().compatibility_status_with(peer_version)

    def is_peer_version_compatible(self, peer_email: str) -> bool:
        """True for SAME and PATCH_DIFF; False for INCOMPATIBLE and UNKNOWN."""
        status = self.peer_compatibility_status(peer_email)
        return status in (CompatibilityStatus.SAME, CompatibilityStatus.PATCH_DIFF)

    def get_peer_compatibility_status(
        self,
        peer_email: str,
        action: CompatAction,
        ignore_peer_version: bool = False,
    ) -> PeerCompatibilityResult:
        """Build a PeerCompatibilityResult describing whether the caller should
        skip / raise / warn for this peer.

        SAME → no skip, no log.

        PATCH_DIFF → no skip and a "patch differs" log, or a skip when
        `skip_peer_on_patch_version_diff` is set.

        INCOMPATIBLE and UNKNOWN → skip, unless effective
        `force_ignore_peer_version or ignore_peer_version`. The UNKNOWN message
        includes a "call client.sync()" hint. A protocol applies its own floor on
        top of this gate, so a peer that passes here can still be refused for one
        protocol and accepted for another.
        """
        own_version = self.get_own_version()
        peer_version = self.get_peer_version(peer_email)
        status = self.peer_compatibility_status(peer_email)
        common = dict(
            peer_email=peer_email,
            status=status,
            action=action,
            suppress_version_warnings=self.suppress_version_warnings,
            local_version=own_version,
            peer_version=peer_version,
        )

        if status == CompatibilityStatus.SAME:
            return PeerCompatibilityResult(should_skip=False, **common)

        if status == CompatibilityStatus.PATCH_DIFF:
            patch_text = (
                own_version.get_patch_warning_text(peer_version)
                if peer_version is not None
                else None
            )
            if self.skip_peer_on_patch_version_diff:
                effective_ignore = self.force_ignore_peer_version or ignore_peer_version
                if effective_ignore:
                    return PeerCompatibilityResult(
                        should_skip=False,
                        explanation_not_skip=(
                            f"Proceeding anyway to {action.value} for peer "
                            f"{peer_email}: {patch_text}."
                        ),
                        **common,
                    )
                return PeerCompatibilityResult(
                    should_skip=True,
                    explanation_skip=f"Skipping peer {peer_email}: {patch_text}.",
                    **common,
                )
            explanation_not_skip = (
                f"Peer {peer_email}: {patch_text}" if patch_text else None
            )
            return PeerCompatibilityResult(
                should_skip=False,
                explanation_not_skip=explanation_not_skip,
                **common,
            )

        # UNKNOWN or INCOMPATIBLE. An unknown peer has no capabilities to check,
        # and an incompatible client version is refused by default. Either way the
        # override lets the caller proceed.
        if status == CompatibilityStatus.UNKNOWN:
            detail = (
                "version information not available "
                "(if you are unsure if it is up to date, call client.sync())"
            )
        else:
            detail = own_version.get_incompatibility_reason(peer_version)

        effective_ignore = self.force_ignore_peer_version or ignore_peer_version
        if effective_ignore:
            return PeerCompatibilityResult(
                should_skip=False,
                explanation_not_skip=(
                    f"Proceeding anyway to {action.value} for peer "
                    f"{peer_email}: {detail}."
                ),
                **common,
            )
        return PeerCompatibilityResult(
            should_skip=True,
            explanation_skip=f"Skipping peer {peer_email}: {detail}.",
            **common,
        )

    def get_compatible_peer_emails_for_syncing(
        self,
        peer_emails: List[str],
        ignore_peer_version: bool = False,
    ) -> List[str]:
        """Filter peer emails to those compatible enough to sync with.

        Uses `get_peer_compatibility_status` and emits any associated warnings
        via `PeerCompatibilityResult.maybe_warn()`.
        """
        compatible: List[str] = []
        for email in peer_emails:
            result = self.get_peer_compatibility_status(
                email,
                action=CompatAction.SYNC,
                ignore_peer_version=ignore_peer_version,
            )
            result.maybe_warn()
            if not result.should_skip:
                compatible.append(email)
        return compatible

    def warn_if_all_peers_incompatible(self, peer_emails: List[str]) -> None:
        """Warn if all connected peers are incompatible (used by DS during sync)."""
        if self.suppress_version_warnings:
            return
        if self.force_ignore_peer_version:
            return
        if not peer_emails:
            return

        any_compatible = any(
            self.peer_compatibility_status(email)
            in (CompatibilityStatus.SAME, CompatibilityStatus.PATCH_DIFF)
            for email in peer_emails
        )
        if not any_compatible:
            warnings.warn(
                f"All connected peers ({len(peer_emails)}) run an incompatible "
                "client version, or their version is unknown. Such a peer is "
                "skipped, and cannot receive jobs or datasets; call client.sync() "
                "to read the version of each peer."
            )

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=False)

    # ========== Peer Management Methods ==========

    def add_peer(self, peer_email: str, force: bool = False, verbose: bool = True):
        """Add a peer — creates peer datasite folders on own drive, shares with peer.

        State-aware and idempotent:
        - If peer is REQUESTED_BY_PEER, creates our folders and marks ACCEPTED.
        - If peer is REQUESTED_BY_ME or ACCEPTED, skips (unless force=True).
        - If no existing peer, creates folders and marks REQUESTED_BY_ME.
        """
        existing_peer_obj = self.get_cached_peer(peer_email)
        if self._return_early(existing_peer_obj, peer_email, force, verbose):
            return

        is_accepting = (
            existing_peer_obj is not None
            and existing_peer_obj.state == PeerState.REQUESTED_BY_PEER
        )
        new_state = PeerState.ACCEPTED if is_accepting else PeerState.REQUESTED_BY_ME
        if verbose:
            self._print_add_peer_start(peer_email, force, is_accepting)

        peer = self.connection_router.add_peer(peer_email=peer_email, verbose=False)

        # Both sides publish their bundle at peering, so a pair can start to
        # encrypt in a later session. The peer's bundle is None until it does.
        self._write_encryption_bundle_for_peer(peer_email)
        peer.info = self.connection_router.read_peer_info_file(peer_email)
        peer.public_encryption_bundle = self._read_peer_encryption_bundle(peer_email)

        self.connection_router.update_peer_state(
            peer_email,
            new_state.value,
            public_encryption_bundle=peer.public_encryption_bundle,
        )
        self.share_peer_info_with_peer(peer_email)

        peer.state = new_state
        self.peer_store.set_peer(peer)

        if verbose:
            self._print_add_peer_done(peer_email, is_accepting)

    def _return_early(
        self, existing_peer_obj, peer_email: str, force: bool, verbose: bool
    ) -> bool:
        """True if the peer is already requested by us or accepted (unless force).

        REQUESTED_BY_PEER returns False so add_peer accepts the incoming request.
        """
        if not existing_peer_obj or force:
            return False
        state = existing_peer_obj.state
        if state not in (PeerState.REQUESTED_BY_ME, PeerState.ACCEPTED):
            return False
        if verbose:
            print_peer_already_connected(peer_email, state.value)
        return True

    def _print_add_peer_start(
        self, peer_email: str, force: bool, is_accepting: bool
    ) -> None:
        if force:
            print_peer_request_resending(peer_email)
        elif is_accepting:
            print_peer_request_accepting(peer_email)
        else:
            print_peer_request_sending(peer_email)

    def _print_add_peer_done(self, peer_email: str, is_accepting: bool) -> None:
        if is_accepting:
            print_peer_connection_established(peer_email)
        else:
            print_peer_request_sent(peer_email)

    def _write_encryption_bundle_for_peer(self, peer_email: str) -> dict | None:
        """Write own encryption bundle for a peer."""
        bundle = self.peer_store.get_public_bundle()
        bundle_json = json.dumps({"public_encryption_bundle": bundle})
        self.connection_router.write_encryption_bundle(peer_email, bundle_json)
        self.connection_router.share_encryption_bundles_folder(peer_email)
        return bundle

    def _read_peer_encryption_bundle(self, peer_email: str) -> dict | None:
        """The bundle ``peer_email`` published for us, or None when it has not published one.

        Raises:
            ValueError: when the published bundle file is not valid JSON.
            KeyError: when it holds no bundle.
        """
        bundle_json = self.connection_router.read_peer_encryption_bundle(peer_email)
        if not bundle_json:
            return None
        return json.loads(bundle_json)["public_encryption_bundle"]

    def load_peers(self, force_download: bool = False):
        """Load peers: from JSON (accepted + requested_by_me) + new requests from folder scan.

        Args:
            force_download: If True, re-fetch SYFT_peers.json from Drive instead
                of using the cached copy. Use when an external writer (e.g. another
                process) may have modified the file.
        """
        if not self.has_do_role and not self.has_ds_role:
            raise ValueError("Client has no role. Set has_do_role or has_ds_role.")

        json_peers = self.connection_router.get_all_peers_from_json(
            force_download=force_download
        )
        peers = [
            p
            for p in json_peers
            if p.state in (PeerState.ACCEPTED, PeerState.REQUESTED_BY_ME)
        ]

        # Detect new peer requests (folders for our datasite not yet in JSON)
        known_emails = {p.email for p in peers}
        peer_request_emails = {
            p.email for p in self.connection_router.get_peer_requests()
        }
        for email in peer_request_emails:
            if email not in known_emails:
                peers.append(
                    Peer(
                        email=email,
                        platforms=[self.connection_router.transport.platform],
                        state=PeerState.REQUESTED_BY_PEER,
                    )
                )
            else:
                # Both sides created folders — upgrade state
                existing = next(p for p in peers if p.email == email)
                if existing.state == PeerState.REQUESTED_BY_ME:
                    existing.state = PeerState.ACCEPTED
                    self.connection_router.update_peer_state(
                        email, PeerState.ACCEPTED.value
                    )

        first_load = not self._first_load_done
        if first_load:
            self._first_load_done = True
            for p in peers:
                if p.state in (PeerState.ACCEPTED, PeerState.REQUESTED_BY_PEER):
                    p.state_at_start = p.state
        self.peer_store.set_peers(peers)
        self.load_peer_infos_parallel([peer.email for peer in peers])

        # Peer policies are known only after the versions load.
        encrypted = [
            p
            for p in peers
            if p.state in (PeerState.ACCEPTED, PeerState.REQUESTED_BY_ME)
            and self.peer_store.pair_encrypts(p)
        ]
        if first_load:
            # Only a pair that encrypts uses the bundle we published.
            self._check_published_bundles([p.email for p in encrypted])
        self._read_missing_peer_bundles(encrypted)

    def _check_published_bundles(self, peer_emails: list[str]) -> None:
        """Raise when a bundle this client published differs from its keys.

        A runtime without the key file of the account makes new keys, and
        peers keep the bundle of the earlier keys. The keys do not change
        while the process runs, so the first peer load checks once.

        Raises:
            ValueError: when a published bundle carries other keys.
        """
        own_bundle = self.peer_store.get_public_bundle()
        published = self._executor.map(self._read_own_published_bundle, peer_emails)
        for peer_email, bundle in zip(peer_emails, published):
            if bundle is not None and bundle != own_bundle:
                raise ValueError(
                    f"The encryption key published for {peer_email} is not the "
                    "key of this client: this account already has a key. Log in "
                    "again with the key file of this account."
                )

    def _read_own_published_bundle(self, peer_email: str) -> dict | None:
        """The bundle this client published for ``peer_email``, or None (thread-safe)."""
        connection = self.connection_router.connection_for_peer_file_read(
            create_new=True
        )
        bundle_json = connection.read_own_encryption_bundle(peer_email)
        if bundle_json is None:
            return None
        return json.loads(bundle_json)["public_encryption_bundle"]

    def _read_missing_peer_bundles(self, peers: list[Peer]) -> None:
        """Read the published bundle of every peer we hold none for."""
        for peer in peers:
            if self.peer_store.has_peer_bundle(peer.email):
                continue
            bundle = self._read_peer_encryption_bundle(peer.email)
            if bundle:
                self.peer_store.set_peer_bundle(peer.email, bundle)
                self.connection_router.update_peer_state(
                    peer.email, peer.state.value, public_encryption_bundle=bundle
                )

    def check_encryption_policy(self) -> None:
        """Raise when the datasite policy on disk is not the policy of this client.

        Another process may have deleted the SyftBox and started with another
        policy. A missing settings file gets the policy of this client.

        Raises:
            EncryptionPolicyMismatchError: if the policy on disk differs.
        """
        resolve_encryption_policy(
            datasite_settings_path(self.syftbox_folder, self.peer_store.email),
            self.peer_store.encryption_policy,
        )

    def validate_peer(self, peer_email: str) -> Peer:
        """Return the peer ``peer_email`` if it is approved, else raise.

        If the peer was already approved, or already waiting for our approval,
        at the first ``load_peers()`` of this client, warn: the connection can
        be left over from an earlier run, with an account that is not in use
        now. This is a setup check for demos, not authentication.

        Raises:
            PeerNotReadyError: the peer did not approve the request yet.
            PeerSetupError: the peer is unknown or rejected, or waits for
                this client to approve its request.
        """
        peer = self.get_cached_peer(peer_email)
        if peer is None:
            raise PeerSetupError(
                f"{peer_email} is not a peer. Check the email, or add it with "
                f"client.add_peer({peer_email!r})."
            )
        if peer.is_requested_by_peer:
            raise PeerSetupError(
                f"{peer_email} is not approved yet: approve it with "
                f"client.approve_peer_request({peer_email!r})."
            )
        if peer.is_requested_by_me:
            raise PeerNotReadyError(
                f"{peer_email} has not approved the request yet. Check the email, "
                "and make sure that the peer approved the request."
            )
        if not peer.is_approved:
            raise PeerSetupError(
                f"{peer_email} is not approved (state: {peer.state.value})."
            )
        self._warn_if_known_at_start(peer_email)
        return peer

    def _warn_if_known_at_start(self, peer_email: str) -> None:
        """Warn if ``peer_email`` was connected, or waiting, at the first load."""
        peer = self.get_cached_peer(peer_email)
        state = peer.state_at_start if peer is not None else None
        if state is None:
            return
        found = (
            "was already connected"
            if state == PeerState.ACCEPTED
            else "sent its request"
        )
        warnings.warn(
            f"{peer_email} {found} before this client started. If this email "
            "is wrong, the connection can be left over from an earlier run."
        )

    def forget_states_at_start(self) -> None:
        """Count no peer as there at start, for the rest of this session.

        Call it after this client deletes its state. Its peers then connect
        again from nothing, so ``validate_peer`` does not warn about them.
        """
        self._first_load_done = True
        for peer in self.peer_store.peers:
            peer.state_at_start = None

    def peer_may_be_valid(self, peer_email: str) -> bool:
        """Tell if ``validate_peer`` can pass after the next ``load_peers()``.

        Use it between polls, after ``validate_peer`` raised
        ``PeerNotReadyError``. It makes one Drive request and does not load
        the peers. It is True when the peer created its folders for this
        datasite, which it does when it approves our request.
        """
        return any(
            p.email == peer_email for p in self.connection_router.get_peer_requests()
        )

    def check_peer_request_exists(self, email: str) -> bool:
        """Check if a peer request exists for the given email."""
        return any(p.email == email for p in self.requested_by_peer_peers)

    def approve_peer_request(
        self,
        email_or_peer: str | Peer,
        verbose: bool = True,
        peer_must_exist: bool = True,
    ):
        """Approve a pending peer request.

        Validates the request exists, then delegates to add_peer(force=True).
        """
        email = email_or_peer if isinstance(email_or_peer, str) else email_or_peer.email

        peer = self.get_cached_peer(email)
        if peer and peer.is_approved:
            if verbose:
                print(f"Peer {email} is already approved, skipping")
            return

        peer_can_approve = peer and peer.state == PeerState.REQUESTED_BY_PEER
        if not peer_can_approve:
            self.load_peers()
            peer = self.get_cached_peer(email)
            peer_can_approve = peer and peer.state == PeerState.REQUESTED_BY_PEER
        if peer_must_exist and not peer_can_approve:
            raise ValueError(
                f"Peer {email} not found in pending requests. "
                f"Use client.peers to see current requests."
            )

        self.add_peer(email, force=True, verbose=False)
        if verbose:
            print(f"Approved peer request from {email}")

    def reject_peer_request(self, email_or_peer: str | Peer):
        """Reject a pending peer request."""
        email = email_or_peer if isinstance(email_or_peer, str) else email_or_peer.email

        peer = self.get_cached_peer(email)
        if not peer or not peer.is_requested_by_peer:
            raise ValueError(
                f"Peer {email} not found in pending requests. "
                f"Use client.peers to see current requests."
            )

        self.connection_router.update_peer_state(email, PeerState.REJECTED.value)
        peer.state = PeerState.REJECTED
        print(f"Rejected peer request from {email}")
