"""PeerStore: merged peer list + encryption key management.

Holds peers, encryption keys, and the encryption policy of this client.
Shared between PeerManager and all ConnectionRouter instances.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import syft_crypto_python as syc
from pydantic import BaseModel, Field, PrivateAttr

from syft.sync.encryption_policy import (
    EncryptionPolicy,
    EncryptionPolicyConflictError,
    MessageFormatError,
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

# Format of the crypto key file. Raise it when the layout of the file changes.
# A file of any other version is refused.
CRYPTO_KEYS_VERSION = 2


def datasite_crypto_keys_path(syftbox_folder: Path | str, email: str) -> Path:
    """Per-datasite key file: ``<syftbox_folder>/<email>/private/crypto_keys.json``."""
    return Path(syftbox_folder) / email / PRIVATE_DIR_NAME / CRYPTO_KEYS_FILENAME


class PeerStore(BaseModel):
    """Manages peers and encryption keys for E2E encryption."""

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    email: str
    encryption_policy: EncryptionPolicy = Field(
        default=EncryptionPolicy.OPTIONAL, frozen=True
    )

    _private_keys: syc.SyftPrivateKeys | None = PrivateAttr(default=None)
    _peers: List[Peer] = PrivateAttr(default_factory=list)
    # The last policy known of each peer in this process (see policy_of).
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
            SyftPeerNotReadyError: when the policy of the peer is not known, or
                the pair encrypts and the peer bundle is missing.
            EncryptionPolicyConflictError: when the policies of the pair conflict.
        """
        self._raise_unless_can_exchange(email)
        if not self.peer_uses_encryption(email):
            return data
        return self.encrypt(email, data)

    def decrypt_and_verify_if_needed(self, email: str, data: bytes) -> bytes:
        """Verify and decrypt a message from ``email`` when the pair encrypts.

        The format of ``data`` must match the pair. Plaintext in a pair that
        encrypts is refused, so that a peer or the storage provider cannot
        downgrade the pair.

        Raises:
            SyftPeerNotReadyError: when the policy of the peer is not known, or
                the pair encrypts and the peer bundle is missing.
            MessageFormatError: when ``data`` is plaintext and the pair
                encrypts, or ``data`` is an envelope and the pair does not.
            EncryptionPolicyConflictError: when the policies of the pair conflict.
        """
        self._raise_unless_can_exchange(email)
        encrypts = self.peer_uses_encryption(email)
        encrypted = self.is_envelope(data)
        if encrypted != encrypts:
            raise MessageFormatError(email, encrypted)
        if not encrypts:
            return data
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
        if policy is None:
            return self._last_known_policies.get(peer.email)
        self._last_known_policies[peer.email] = policy
        return policy

    def conflict_error(self, peer: Peer) -> EncryptionPolicyConflictError | None:
        """Why this client and ``peer`` cannot exchange messages, or None when they can.

        That is when one side has encryption off and the other requires it.
        """
        if pair_conflicts(self.encryption_policy, self.policy_of(peer)):
            return EncryptionPolicyConflictError(peer.email)
        return None

    def _raise_unless_can_exchange(self, email: str) -> None:
        """Raise unless the policy of ``email`` is known and does not conflict.

        Raises:
            EncryptionPolicyConflictError: when the policies of the pair conflict.
            SyftPeerNotReadyError: when the policy of the peer is not known.
        """
        peer = self.get_cached_peer(email)
        error = None if peer is None else self.conflict_error(peer)
        if error is not None:
            raise error
        if peer is None or self.policy_of(peer) is None:
            raise SyftPeerNotReadyError(
                email,
                cause=f"The encryption policy of {email} is not known yet.",
                remedy=(
                    "Peer with them and wait until they accept. Then run "
                    "client.sync() and send again."
                ),
            )

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
        """Encrypt for self if this client requires encryption."""
        if self.use_encryption:
            return self.encrypt_for_self(data)
        return data

    def decrypt_and_verify_for_self_if_needed(self, data: bytes) -> bytes:
        """Verify and decrypt data for self if this client requires encryption."""
        if self.use_encryption:
            self.verify_message_from_self(data)
            return self.decrypt_for_self(data)
        return data

    # ========== Persistence ==========

    def save_keys(self, path: Path | str) -> None:
        """Write our keys to a new key file at ``path``.

        Raises:
            FileExistsError: if a file exists at ``path``.
        """
        keys = self._ensure_private_keys()
        data = {
            "version": CRYPTO_KEYS_VERSION,
            "email": self.email,
            "keys_jwk": keys.to_jwks(),
        }
        _create_key_file(Path(path), data)

    @classmethod
    def create(
        cls,
        email: str,
        keys_path: Path | str,
        encryption_policy: EncryptionPolicy = EncryptionPolicy.OPTIONAL,
    ) -> "PeerStore":
        """Build a PeerStore for ``email`` with the keys in its key file.

        ``keys_path`` is the participant's own key file
        (``<syftbox_folder>/<email>/private/crypto_keys.json``, see
        :func:`datasite_crypto_keys_path`). When the file exists, the store
        loads its keys. Else the store generates keys and writes the file.

        Scoping keys to a per-datasite file lets several identities run on one
        machine (e.g. two data owners in the same notebook) without colliding, and
        ties key lifetime to the datasite folder.

        Raises:
            ValueError: if the key file has another version or belongs to
                another identity.
            FileExistsError: if another process created the key file while
                this store generated its keys.
        """
        store = cls(email=email, encryption_policy=encryption_policy)
        path = Path(keys_path)
        if path.exists():
            store._private_keys = _read_private_keys(path, email)
        else:
            store.generate_keys()
            store.save_keys(path)
        return store


def import_key_file(source: Path | str, keys_path: Path | str, email: str) -> None:
    """Copy the key file ``source`` of ``email`` to ``keys_path``.

    Use it to restore the keys of an earlier session, e.g. an uploaded key file
    on Colab, before the client is built. When ``keys_path`` holds the same
    keys already, nothing changes.

    Raises:
        ValueError: if ``source`` has another version or belongs to another
            identity.
        FileExistsError: if ``keys_path`` holds other keys.
    """
    source, keys_path = Path(source), Path(keys_path)
    _read_private_keys(source, email)
    data = json.loads(source.read_text())
    if keys_path.exists():
        if json.loads(keys_path.read_text()) != data:
            raise FileExistsError(
                f"{keys_path} holds other keys than {source}. Remove it to "
                "import the file, or log in without the file."
            )
        return
    _create_key_file(keys_path, data)


def _read_private_keys(path: Path, email: str) -> syc.SyftPrivateKeys:
    data = json.loads(path.read_text())
    if data["version"] != CRYPTO_KEYS_VERSION:
        raise ValueError(
            f"The crypto key file at {path} has version {data['version']}, and "
            f"this client reads only version {CRYPTO_KEYS_VERSION}."
        )
    if data["email"] != email:
        raise ValueError(
            f"Encryption key file {path} belongs to {data['email']!r}, not {email!r}"
        )
    return syc.SyftPrivateKeys.from_jwks(data["keys_jwk"])


def _create_key_file(path: Path, data: dict) -> None:
    """Write ``data`` to a new key file at ``path`` in one step.

    The key file holds the only copy of the private keys. The data goes to a
    temporary file in the same folder and reaches the disk. A hard link then
    gives it the name of the key file. The link fails when that name exists,
    so a write cut short leaves no partial file, and a second process cannot
    replace the keys of the first. ``mkstemp`` makes the file readable by this
    user only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
