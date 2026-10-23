"""PeerStore: merged peer list + encryption key management.

Holds peers, encryption keys, and the encryption policy of this client.
Shared between PeerManager and all ConnectionRouter instances.
"""

import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import portalocker
import syft_crypto_python as syc
from pydantic import BaseModel, PrivateAttr

from syft.sync.encryption_policy import (
    EncryptionPolicy,
    EncryptionPolicyConflictError,
    PeerEncryptionPolicyUnknownError,
    PeerKeysNotReadyError,
    PeerOutdatedError,
    UnencryptedMessageError,
    pair_conflicts,
    pair_uses_encryption,
)
from syft.sync.peers.exceptions import SyftPeerNotReadyError
from syft.sync.peers.peer import Peer, PeerState

# Encryption key bundles persist inside the participant's own SyftBox datasite
# folder, under private/ (which is never synced to Drive). This scopes keys per
# identity by location, so several identities can run on one machine without
# colliding, and delete_syftbox removes them together with the folder.
PRIVATE_DIR_NAME = "private"
CRYPTO_KEYS_FILENAME = "crypto_keys.json"

# Format of the crypto key file. Raise it when the layout of the file changes,
# and add a read path for every earlier version. A file with no version was
# written before the field, and is version 0.
CRYPTO_KEYS_VERSION = 1
# Key in the crypto key file that records the policy. Every process that
# loads the file (a notebook and syft-bg) then uses the same policy. A file
# without it was written when only required clients had keys.
ENCRYPTION_POLICY_KEY = "encryption_policy"


def datasite_crypto_keys_path(syftbox_folder: Path | str, email: str) -> Path:
    """Per-datasite key file: ``<syftbox_folder>/<email>/private/crypto_keys.json``."""
    return Path(syftbox_folder) / email / PRIVATE_DIR_NAME / CRYPTO_KEYS_FILENAME


class PeerStore(BaseModel):
    """Manages peers and encryption keys for E2E encryption."""

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    email: str
    encryption_policy: EncryptionPolicy = EncryptionPolicy.OPTIONAL

    _private_keys: syc.SyftPrivateKeys | None = PrivateAttr(default=None)
    _peers: List[Peer] = PrivateAttr(default_factory=list)
    # The private key file. None when the store was built in memory, e.g. in
    # tests.
    _keys_path: Path | None = PrivateAttr(default=None)
    # Peers whose version file was read again since the last ``set_peers``.
    _policy_rereads: set[str] = PrivateAttr(default_factory=set)
    # The last policy known of each peer in this process (see peer_policy).
    _last_known_policies: dict[str, EncryptionPolicy] = PrivateAttr(
        default_factory=dict
    )

    @property
    def use_encryption(self) -> bool:
        """True when this client requires encryption with every peer and at rest."""
        return self.encryption_policy == EncryptionPolicy.REQUIRED

    # ========== Peer list methods ==========

    def clear_caches(self) -> None:
        """Clear the caches."""
        self._peers = []

    @property
    def approved_peers(self) -> List[Peer]:
        return [p for p in self._peers if p.is_approved]

    @property
    def requested_by_peer_peers(self) -> List[Peer]:
        return [p for p in self._peers if p.is_requested_by_peer]

    @property
    def requested_by_me_peers(self) -> List[Peer]:
        return [p for p in self._peers if p.is_requested_by_me]

    @property
    def syncable_peers(self) -> List[Peer]:
        return [p for p in self._peers if p.is_requested_by_me or p.is_approved]

    def get_cached_peer(self, email: str) -> Optional[Peer]:
        for p in self._peers:
            if p.email == email:
                return p
        return None

    def encrypt_if_needed(self, email: str, data: bytes) -> bytes:
        """Encrypt ``data`` for ``email`` when the pair encrypts.

        Raises:
            PeerEncryptionPolicyUnknownError: when this client is optional and
                the policy of the peer is not known.
            PeerKeysNotReadyError: when the pair encrypts, but our keys or the
                peer bundle are missing.
            EncryptionPolicyConflictError: when the policies of the pair conflict.
        """
        error = self._cached_conflict_error(email)
        if error is not None:
            raise error
        if not self.peer_policy_known(email):
            raise PeerEncryptionPolicyUnknownError(email)
        if not self.peer_uses_encryption(email):
            return data
        if not self.encryption_ready(email):
            raise PeerKeysNotReadyError(email)
        return self.encrypt(email, data)

    def decrypt_and_verify_if_needed(self, email: str, data: bytes) -> bytes:
        """Verify and decrypt an envelope from ``email``; pass plaintext through.

        When the pair encrypts, plaintext is refused, so that a peer or the
        storage provider cannot downgrade the pair. While the peer policy is
        not known, plaintext waits, as a send does. When the pair does not
        encrypt, an envelope is still accepted: the peer can have started to
        require encryption after the last load.

        Raises:
            PeerEncryptionPolicyUnknownError: when ``data`` is plaintext and
                the peer policy is not known.
            UnencryptedMessageError: when the pair encrypts and ``data`` is
                plaintext.
            PeerKeysNotReadyError: when ``data`` is an envelope, but our keys
                or the peer bundle are missing.
            EncryptionPolicyConflictError: when the policies of the pair conflict.
        """
        error = self._cached_conflict_error(email)
        if error is not None:
            raise error
        if not self.is_envelope(data):
            if not self.peer_policy_known(email):
                raise PeerEncryptionPolicyUnknownError(email)
            if self.peer_uses_encryption(email):
                raise UnencryptedMessageError(email)
            return data
        if not self.encryption_ready(email):
            raise PeerKeysNotReadyError(email)
        self.verify_message(email, data)
        return self.decrypt(email, data)

    @staticmethod
    def is_envelope(data: bytes) -> bool:
        """True if `data` is an SYC encryption envelope (vs plaintext)."""
        try:
            syc.parse_envelope(data)
            return True
        except Exception:
            return False

    def decrypt_dataset_if_needed(self, owner_email: str, data: bytes) -> bytes:
        """Decrypt a downloaded dataset file, tolerating plaintext.

        Dataset *collections* (public mock previews) are uploaded unencrypted, so
        plaintext bytes are passed through instead of raising. Bytes that are an
        SYC envelope are still signature-verified and decrypted as usual.
        """
        if not self.is_envelope(data):
            return data
        self.verify_message(owner_email, data)
        return self.decrypt(owner_email, data)

    def peer_policy(self, email: str) -> Optional[EncryptionPolicy]:
        """The policy of the cached peer ``email``, or None when not known."""
        peer = self.get_cached_peer(email)
        return None if peer is None else self.policy_of(peer)

    def policy_of(self, peer: Peer) -> Optional[EncryptionPolicy]:
        """The policy of ``peer``, or None when not known.

        When the version file can no longer be read (the peer deleted its
        SyftBox, or a read failed), the last policy known in this process
        applies.
        """
        policy = peer.encryption_policy
        if policy is None and not peer.is_outdated:
            return self._last_known_policies.get(peer.email)
        self._last_known_policies[peer.email] = policy
        return policy

    def peer_policy_known(self, email: str) -> bool:
        """Whether this client can select the format for ``email``.

        A required client encrypts with every peer, so it needs no peer policy.
        """
        return self.use_encryption or self.peer_policy(email) is not None

    def encryption_ready(self, email: str) -> bool:
        """Whether our keys and the bundle of ``email`` exist."""
        return self.has_my_keys() and self.has_peer_bundle(email)

    def conflict_error(self, peer: Peer) -> EncryptionPolicyConflictError | None:
        """Why this client and ``peer`` cannot exchange messages, or None when they can.

        That is when one side has encryption off and the other requires it,
        or when the peer runs an outdated syft.
        """
        if peer.is_outdated:
            return PeerOutdatedError(peer.email)
        if pair_conflicts(self.encryption_policy, self.policy_of(peer)):
            return EncryptionPolicyConflictError(peer.email)
        return None

    def _cached_conflict_error(
        self, email: str
    ) -> EncryptionPolicyConflictError | None:
        peer = self.get_cached_peer(email)
        return None if peer is None else self.conflict_error(peer)

    def peer_uses_encryption(self, email: str) -> bool:
        """Whether the pair with ``email`` encrypts. False when the peer policy is not known."""
        return pair_uses_encryption(self.encryption_policy, self.peer_policy(email))

    def set_peer(self, peer: Peer) -> None:
        for i, p in enumerate(self._peers):
            if p.email == peer.email:
                self._peers[i] = peer
                return
        self._peers.append(peer)

    def add_peer(self, peer: Peer) -> None:
        self._peers.append(peer)

    def set_peers(self, peers: List[Peer]) -> None:
        self._peers = peers
        self._policy_rereads.clear()

    def should_hold(self, recipient: str) -> bool:
        """Whether a message to ``recipient`` waits while encryption is pending.

        Only a peer waits. Any other recipient never gets a policy, so a
        message to it would wait forever.
        """
        return self.get_cached_peer(recipient) is not None

    def take_policy_reread(self, email: str) -> bool:
        """True once per peer load: the caller may read the version file again."""
        if email in self._policy_rereads:
            return False
        self._policy_rereads.add(email)
        return True

    # ========== Ensure helpers ==========

    def _ensure_private_keys(self) -> syc.SyftPrivateKeys:
        if self._private_keys is None:
            raise ValueError(
                f"No private keys for {self.email}. Encryption is on for this "
                "client, but it holds no key pair. Keys are created at login, "
                "so log in again to create them."
            )
        return self._private_keys

    def _ensure_peer(self, email: str) -> Peer:
        peer = self.get_cached_peer(email)
        if peer is None:
            raise SyftPeerNotReadyError(
                email,
                cause=f"No cached peer for {email}.",
                remedy=(
                    "Run client.sync() to refresh your peers, or "
                    f"client.add_peer('{email}') if you have not added them."
                ),
            )
        return peer

    def _ensure_peer_bundle(self, email: str) -> dict:
        peer = self._ensure_peer(email)
        if peer.public_encryption_bundle is None:
            cause, remedy = self._missing_bundle_reason(peer)
            raise SyftPeerNotReadyError(email, cause=cause, remedy=remedy)
        return peer.public_encryption_bundle

    @staticmethod
    def _missing_bundle_reason(peer: Peer) -> tuple[str, str]:
        """The cause and the remedy for a peer that has no encryption bundle.

        A bundle is only stored for a peer who is accepted (or requested by
        you) and who has already published one, so the four states below are
        the four reasons the bundle can be missing.
        """
        email = peer.email
        if peer.state == PeerState.REQUESTED_BY_PEER:
            return (
                f"{email} asked to peer with you, but you have not accepted yet.",
                f"Run client.approve_peer_request('{email}').",
            )
        if peer.state == PeerState.REJECTED:
            return (
                f"You rejected {email}, so no encryption bundle was kept.",
                f"Run client.add_peer('{email}') to peer with them again.",
            )
        if peer.state == PeerState.REQUESTED_BY_ME:
            return (
                f"You asked to peer with {email}, but they have not accepted yet.",
                "Wait for them to accept, then run client.sync().",
            )
        return (
            f"{email} is an accepted peer, but you do not have their public "
            "encryption bundle. Either they have not published one yet, or you "
            "have not synced since they did.",
            "Run client.sync(). If it continues, the peer may not have "
            "encryption enabled.",
        )

    # ========== Crypto methods ==========

    def generate_keys(self) -> None:
        self._private_keys = syc.SyftRecoveryKey.generate().derive_keys()

    def ensure_keys(self) -> None:
        """Make sure this client has keys: keep them, load the key file, or generate.

        An optional client has no keys until a peer requires encryption. New
        keys are saved to the key file when the store has one. The lock covers
        the check and the write, so two processes on one datasite (a notebook
        and syft-bg) do not each make a key pair.
        """
        if self.has_my_keys():
            return
        if self._keys_path is None:
            self.generate_keys()
            return
        with _locked_key_file(self._keys_path):
            if self._keys_path.exists():
                self._adopt_key_file(self.load_keys(self._keys_path))
            else:
                self.generate_keys()
                _write_key_file(self._keys_path, self._key_file_data())

    def check_key_file_policy(self) -> None:
        """Raise when the key file requires encryption and this process does not.

        The policy is read at startup only. Another process (a new login) can
        make the datasite required later; this process must not go on
        without encryption.

        Raises:
            ValueError: when the key file records a required policy, and this
                client is optional or off.
        """
        if self.use_encryption or self._keys_path is None:
            return
        data = self._read_key_file()
        if data is None:
            return
        recorded = data.get(ENCRYPTION_POLICY_KEY, EncryptionPolicy.REQUIRED.value)
        if recorded == EncryptionPolicy.REQUIRED.value:
            raise ValueError(
                f"The key file {self._keys_path} requires encryption, but this "
                f"process started with encryption {self.encryption_policy.value}. "
                "Restart it (for example syft-bg) to use the new policy."
            )

    def _adopt_key_file(self, loaded: "PeerStore") -> None:
        """Take the keys of ``loaded``, a store read from our key file.

        The peers of ``loaded`` are taken only when this store has none.
        """
        if loaded.email != self.email:
            raise ValueError(
                f"Encryption key file {self._keys_path} belongs to "
                f"{loaded.email!r}, not {self.email!r}"
            )
        self._private_keys = loaded._private_keys
        if not self._peers:
            self._peers = loaded._peers

    def has_my_keys(self) -> bool:
        return self._private_keys is not None

    @property
    def public_key(self) -> syc.SyftPublicKeyBundle:
        keys = self._ensure_private_keys()
        return keys.to_public_bundle()

    def get_public_bundle(self) -> dict:
        keys = self._ensure_private_keys()
        bundle = keys.to_public_bundle()
        did = f"did:syft:{self.email}"
        did_doc = bundle.to_did_document(did)
        did_doc["identity"] = self.email
        return did_doc

    def set_peer_bundle(self, peer_email: str, bundle: dict) -> None:
        peer = self._ensure_peer(peer_email)
        peer.public_encryption_bundle = bundle

    def has_peer_bundle(self, peer_email: str) -> bool:
        peer = self.get_cached_peer(peer_email)
        return peer is not None and peer.public_encryption_bundle is not None

    def _get_parsed_peer_bundle(self, peer_email: str) -> syc.SyftPublicKeyBundle:
        bundle = self._ensure_peer_bundle(peer_email)
        return syc.SyftPublicKeyBundle.from_did_document(bundle)

    def verify_message(self, sender_email: str, envelope: bytes) -> None:
        """Verify the envelope signature against the sender's public key. Raises on failure."""
        sender_bundle = self._get_parsed_peer_bundle(sender_email)
        parsed = syc.parse_envelope(envelope)
        syc.verify_envelope_signature(parsed, sender_bundle.identity_key_bytes)

    def verify_message_from_self(self, envelope: bytes) -> None:
        """Verify the envelope signature against own public key. Raises on failure."""
        keys = self._ensure_private_keys()
        own_bundle = keys.to_public_bundle()
        parsed = syc.parse_envelope(envelope)
        syc.verify_envelope_signature(parsed, own_bundle.identity_key_bytes)

    def encrypt(self, recipient_email: str, plaintext: bytes) -> bytes:
        keys = self._ensure_private_keys()
        peer_bundle = self._get_parsed_peer_bundle(recipient_email)
        recipient = syc.EncryptionRecipient(recipient_email, peer_bundle)
        return syc.encrypt_message(self.email, keys, [recipient], plaintext)

    def decrypt(self, sender_email: str, envelope: bytes) -> bytes:
        keys = self._ensure_private_keys()
        sender_bundle = self._get_parsed_peer_bundle(sender_email)
        parsed = syc.parse_envelope(envelope)
        return syc.decrypt_message(self.email, keys, sender_bundle, parsed)

    # ========== Self-encryption (DO at-rest) ==========

    def encrypt_for_self(self, plaintext: bytes) -> bytes:
        """Encrypt data using own keys with self as recipient."""
        keys = self._ensure_private_keys()
        own_bundle = keys.to_public_bundle()
        recipient = syc.EncryptionRecipient(self.email, own_bundle)
        return syc.encrypt_message(self.email, keys, [recipient], plaintext)

    def decrypt_for_self(self, envelope: bytes) -> bytes:
        """Decrypt data that was encrypted for self."""
        keys = self._ensure_private_keys()
        own_bundle = keys.to_public_bundle()
        parsed = syc.parse_envelope(envelope)
        return syc.decrypt_message(self.email, keys, own_bundle, parsed)

    def encrypt_for_self_if_needed(self, data: bytes) -> bytes:
        """Encrypt for self if encryption is enabled and keys are available."""
        if self.use_encryption and self.has_my_keys():
            return self.encrypt_for_self(data)
        return data

    def decrypt_and_verify_for_self_if_needed(self, data: bytes) -> bytes:
        """Verify and decrypt an envelope this client wrote for itself; pass plaintext through.

        Files written while this client was optional stay plaintext.
        """
        if self.is_envelope(data):
            self.verify_message_from_self(data)
            return self.decrypt_for_self(data)
        return data

    # ========== Persistence ==========

    def _key_file_data(self) -> dict:
        """The key file as this store would write it from memory."""
        keys = self._ensure_private_keys()
        return {
            "version": CRYPTO_KEYS_VERSION,
            "email": self.email,
            "keys_jwk": keys.to_jwks(),
            "peer_bundles": {
                peer.email: peer.public_encryption_bundle
                for peer in self._peers
                if peer.public_encryption_bundle is not None
            },
            ENCRYPTION_POLICY_KEY: self.encryption_policy.value,
        }

    def _record_required_policy(self) -> None:
        """Record a required policy in the key file, keeping what else it holds."""
        if self._keys_path is None or not self.use_encryption:
            return
        with _locked_key_file(self._keys_path):
            data = self._read_key_file() or self._key_file_data()
            if data.get(ENCRYPTION_POLICY_KEY) == EncryptionPolicy.REQUIRED.value:
                return
            data = {**data, ENCRYPTION_POLICY_KEY: EncryptionPolicy.REQUIRED.value}
            _write_key_file(self._keys_path, data)
        _warn_key_file_holds_data(self._keys_path)

    def save_keys(self, path: Path) -> None:
        data = self._key_file_data()
        path = Path(path)
        with _locked_key_file(path):
            _write_key_file(path, data)
        self._keys_path = path

    def _read_key_file(self) -> dict | None:
        """The key file as a dict, or None when missing, unreadable or another identity's."""
        if self._keys_path is None:
            return None
        try:
            data = json.loads(self._keys_path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("email") != self.email:
            return None
        return data

    @classmethod
    def load_keys(cls, path: Path) -> "PeerStore":
        data = json.loads(Path(path).read_text())
        # A file with no version was written before the field, and its layout is
        # this client reads. A later version is refused: a user cannot rebuild a
        # private key, so a wrong read loses the keys.
        version = data.get("version", 0)
        if version > CRYPTO_KEYS_VERSION:
            raise ValueError(
                f"The crypto key file at {path} has version {version}, and this "
                f"client reads up to version {CRYPTO_KEYS_VERSION}. Install a "
                "newer syft to use these keys."
            )
        policy = data.get(ENCRYPTION_POLICY_KEY, EncryptionPolicy.REQUIRED.value)
        store = cls(email=data["email"], encryption_policy=EncryptionPolicy(policy))
        store._private_keys = syc.SyftPrivateKeys.from_jwks(data["keys_jwk"])
        store._keys_path = Path(path)
        for email, bundle_dict in data.get("peer_bundles", {}).items():
            store._peers.append(Peer(email=email, public_encryption_bundle=bundle_dict))
        return store

    @classmethod
    def create(
        cls,
        email: str,
        encryption_policy: EncryptionPolicy = EncryptionPolicy.OPTIONAL,
        keys_path: Path | str | None = None,
    ) -> "PeerStore":
        """Build a PeerStore for ``email`` with its per-datasite key file.

        ``keys_path`` is the participant's own key file
        (``<syftbox_folder>/<email>/private/crypto_keys.json``, see
        :func:`datasite_crypto_keys_path`). When the file exists, the store
        loads its keys and peer bundles. A required policy in the file wins over an
        optional ``encryption_policy``, so all processes on a datasite agree.

        Without a file, a required store generates keys and saves them. An
        optional store has no keys until :meth:`ensure_keys`.

        Scoping keys to a per-datasite file lets several identities run on one
        machine (e.g. two data owners in the same notebook) without colliding, and
        ties key lifetime to the datasite folder.

        Raises:
            ValueError: if the policy is required but ``keys_path`` is missing,
                or the existing key file belongs to a different identity.
        """
        store = cls(email=email, encryption_policy=encryption_policy)
        if keys_path is None:
            if store.use_encryption:
                raise ValueError("keys_path is required when encryption is required")
            return store
        store._keys_path = Path(keys_path)
        if store._keys_path.exists():
            loaded = cls.load_keys(store._keys_path)
            if encryption_policy == EncryptionPolicy.OFF and loaded.use_encryption:
                raise _off_refused_error(store._keys_path)
            if loaded.use_encryption:
                store.encryption_policy = EncryptionPolicy.REQUIRED
            store._adopt_key_file(loaded)
            store._record_required_policy()
        elif store.use_encryption:
            store.generate_keys()
            store.save_keys(store._keys_path)
            _warn_key_file_holds_data(store._keys_path)
        return store


def _off_refused_error(keys_path: Path) -> ValueError:
    return ValueError(
        f"The key file {keys_path} requires encryption: files of this "
        "datasite at rest are encrypted. Encryption cannot be off for it. Use a "
        "new SyftBox folder for a client with encryption off."
    )


def _warn_key_file_holds_data(keys_path: Path) -> None:
    """Warn once, when a datasite starts to require encryption."""
    warnings.warn(
        "This datasite now requires encryption, and every later session keeps "
        "that setting. Its new files at rest are encrypted with the key in "
        f"{keys_path}. Without that file, they cannot be read again: keep a "
        "copy in a safe place."
    )


@contextmanager
def _locked_key_file(path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock on the key file at ``path``.

    The lock is on a sibling ``.lock`` file, as in ``PersistedDict``, so the
    key file itself can be replaced while the lock is held.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(f"{path.suffix}.lock"), "a") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        try:
            yield
        finally:
            portalocker.unlock(f)


def _write_key_file(path: Path, data: dict) -> None:
    """Replace the key file at ``path`` with ``data`` in one step.

    The key file holds the only copy of the private keys, so a write cut short
    must leave the old file in place. The data goes to a temporary file in the
    same folder, reaches the disk, and then takes the name of the key file.
    ``mkstemp`` creates the temporary file readable by this user only.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
