"""PeerStore: merged peer list + encryption key management.

Holds peers, encryption keys, and a use_encryption flag.
Shared between PeerManager and all ConnectionRouter instances.
"""

import json
import logging
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import portalocker
import syft_crypto_python as syc
from pydantic import BaseModel, PrivateAttr

from syft.sync.peers.key_bundle import (
    InvalidPeerBundleError,
    PeerKeyChangedError,
    bundle_fingerprint,
    did_for_email,
    format_fingerprint,
    parse_and_validate_bundle,
)
from syft.sync.peers.peer import Peer

logger = logging.getLogger(__name__)

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
# Key in the crypto key file under which pinned peer bundles are stored.
PEER_BUNDLES_KEY = "peer_bundles"


def datasite_crypto_keys_path(syftbox_folder: Path | str, email: str) -> Path:
    """Per-datasite key file: ``<syftbox_folder>/<email>/private/crypto_keys.json``."""
    return Path(syftbox_folder) / email / PRIVATE_DIR_NAME / CRYPTO_KEYS_FILENAME


class PeerStore(BaseModel):
    """Manages peers and encryption keys for E2E encryption.

    Peer public key bundles are *pinned*: the first validated bundle seen for a
    peer is kept, persisted next to our own keys in the private key file, and a
    later bundle with a different identity key is refused unless the caller
    passes ``allow_key_change=True``. The pin lives in the local key file, not
    in ``SYFT_peers.json`` on Drive, so an edit to the Drive copy cannot swap
    the key we encrypt to.
    """

    model_config = {"arbitrary_types_allowed": True}

    email: str
    use_encryption: bool = False

    _private_keys: syc.SyftPrivateKeys | None = PrivateAttr(default=None)
    _peers: List[Peer] = PrivateAttr(default_factory=list)
    # Where pinned peer bundles persist (the private key file). None when the
    # store was built in memory, e.g. in tests; pins then live only in memory.
    _keys_path: Path | None = PrivateAttr(default=None)

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
        if self.peer_uses_encryption(email):
            return self.encrypt(email, data)
        return data

    def decrypt_and_verify_if_needed(self, email: str, data: bytes) -> bytes:
        if self.peer_uses_encryption(email):
            self.verify_message(email, data)
            return self.decrypt(email, data)
        return data

    def _is_syc_envelope(self, data: bytes) -> bool:
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
        if not self.peer_uses_encryption(owner_email):
            return data
        if not self._is_syc_envelope(data):
            return data
        self.verify_message(owner_email, data)
        return self.decrypt(owner_email, data)

    def peer_uses_encryption(self, email: str) -> bool:
        peer = self.get_cached_peer(email)
        return peer is not None and peer.use_encryption

    def set_peer(self, peer: Peer) -> None:
        peer.use_encryption = self.use_encryption
        self._keep_pinned_bundle(peer)
        for i, p in enumerate(self._peers):
            if p.email == peer.email:
                self._peers[i] = peer
                return
        self._peers.append(peer)

    def add_peer(self, peer: Peer) -> None:
        peer.use_encryption = self.use_encryption
        self._keep_pinned_bundle(peer)
        self._peers.append(peer)

    def set_peers(self, peers: List[Peer]) -> None:
        for p in peers:
            p.use_encryption = self.use_encryption
            self._keep_pinned_bundle(p)
        self._peers = peers

    def _keep_pinned_bundle(self, incoming: Peer) -> None:
        """Make ``incoming`` carry the pinned bundle for its email, if any.

        Peers loaded from ``SYFT_peers.json`` carry whatever bundle the Drive
        copy holds. A pin always wins over that copy: a differing Drive bundle
        is reported and dropped. Without a pin, a valid Drive bundle becomes the
        pin (trust on first use) and an invalid one is dropped.
        """
        if not self.use_encryption:
            return
        pinned = self._current_pin(incoming.email)
        candidate = incoming.public_encryption_bundle
        if pinned is not None:
            if candidate is not None and candidate != pinned:
                if self._is_trusted_replacement(incoming.email, pinned, candidate):
                    self._persist_peer_bundle(incoming.email, candidate)
                    return
                self._warn_ignoring_cached_key(incoming.email, pinned, candidate)
            incoming.public_encryption_bundle = pinned
        elif candidate is not None:
            try:
                parse_and_validate_bundle(incoming.email, candidate)
            except InvalidPeerBundleError as e:
                warnings.warn(f"Dropping cached encryption key: {e}")
                incoming.public_encryption_bundle = None
            else:
                self._persist_peer_bundle(incoming.email, candidate)

    def _current_pin(self, peer_email: str) -> dict | None:
        """The pin for ``peer_email``: the one in memory, else the one in the key file.

        The in-memory list is rebuilt on every sync, so a peer missing from one
        sync (rejected for a while), or pinned by another process, has no pin in
        memory. The key file still holds it, and it must win, or a first use
        would write a Drive bundle over it.
        """
        cached = self.get_cached_peer(peer_email)
        pinned = cached.public_encryption_bundle if cached else None
        if pinned is None:
            pinned = self._valid_stored_pin(peer_email)
        return pinned

    @staticmethod
    def _warn_ignoring_cached_key(
        peer_email: str, pinned: dict, candidate: dict
    ) -> None:
        try:
            over = f" over {format_fingerprint(bundle_fingerprint(candidate))}"
        except ValueError:
            over = ""
        warnings.warn(
            f"Ignoring a cached encryption key for {peer_email} that differs from "
            f"the pinned key. Keeping the pinned key "
            f"{format_fingerprint(bundle_fingerprint(pinned))}{over}."
        )

    def _is_trusted_replacement(
        self, peer_email: str, pinned: dict, candidate: dict
    ) -> bool:
        """Whether ``candidate`` may replace ``pinned`` without a user decision.

        True when it validates and either carries the same identity key (the
        peer re-signed their prekeys; the identity key vouches for them) or
        matches the pin another process on this datasite already recorded in
        the key file (syft-bg trusted it next to a notebook).
        """
        try:
            parsed = parse_and_validate_bundle(peer_email, candidate)
        except InvalidPeerBundleError:
            return False
        if parsed.identity_fingerprint() == bundle_fingerprint(pinned):
            return True
        return candidate == self._stored_pin(peer_email)

    # ========== Ensure helpers ==========

    def _ensure_private_keys(self) -> syc.SyftPrivateKeys:
        if self._private_keys is None:
            raise ValueError("No private keys — call generate_keys() first")
        return self._private_keys

    def _ensure_peer(self, email: str) -> Peer:
        peer = self.get_cached_peer(email)
        if peer is None:
            raise ValueError(f"No cached peer for {email}")
        return peer

    def _ensure_peer_bundle(self, email: str) -> dict:
        peer = self._ensure_peer(email)
        if peer.public_encryption_bundle is None:
            raise ValueError(f"No public encryption bundle for {email}")
        return peer.public_encryption_bundle

    # ========== Crypto methods ==========

    def generate_keys(self) -> None:
        self._private_keys = syc.SyftRecoveryKey.generate().derive_keys()

    def has_my_keys(self) -> bool:
        return self._private_keys is not None

    @property
    def public_key(self) -> syc.SyftPublicKeyBundle:
        keys = self._ensure_private_keys()
        return keys.to_public_bundle()

    def get_public_bundle(self) -> dict:
        keys = self._ensure_private_keys()
        bundle = keys.to_public_bundle()
        did_doc = bundle.to_did_document(did_for_email(self.email))
        did_doc["identity"] = self.email
        return did_doc

    @property
    def my_fingerprint(self) -> str:
        """Fingerprint of our own identity key, to hand to peers out of band."""
        return self.public_key.identity_fingerprint()

    def peer_fingerprint(self, peer_email: str) -> Optional[str]:
        """Fingerprint of the pinned identity key of ``peer_email``, or None."""
        peer = self.get_cached_peer(peer_email)
        if peer is None or peer.public_encryption_bundle is None:
            return None
        return bundle_fingerprint(peer.public_encryption_bundle)

    def validate_peer_bundle(
        self, peer_email: str, bundle: dict
    ) -> syc.SyftPublicKeyBundle:
        """Parse ``bundle``, verify its signatures and its asserted identity.

        Raises:
            InvalidPeerBundleError: when the bundle fails any check.
        """
        return parse_and_validate_bundle(peer_email, bundle)

    def set_peer_bundle(
        self, peer_email: str, bundle: dict, allow_key_change: bool = False
    ) -> None:
        """Pin ``bundle`` as the public key of ``peer_email``.

        The bundle is validated first. When a different key is already pinned
        the call raises :class:`PeerKeyChangedError`, unless
        ``allow_key_change`` is set by a caller acting on an explicit user
        decision (approving a peer, or ``trust_peer_key``).
        """
        peer = self._ensure_peer(peer_email)
        parsed = self.validate_peer_bundle(peer_email, bundle)
        pinned = self._current_pin(peer_email)
        if pinned is not None:
            old_fp = bundle_fingerprint(pinned)
            new_fp = parsed.identity_fingerprint()
            if old_fp != new_fp and not allow_key_change:
                raise PeerKeyChangedError(peer_email, old_fp, new_fp)
            if pinned == bundle:
                peer.public_encryption_bundle = bundle
                return
        peer.public_encryption_bundle = bundle
        self._persist_peer_bundle(peer_email, bundle)

    def has_peer_bundle(self, peer_email: str) -> bool:
        peer = self.get_cached_peer(peer_email)
        return peer is not None and peer.public_encryption_bundle is not None

    def _get_parsed_peer_bundle(self, peer_email: str) -> syc.SyftPublicKeyBundle:
        bundle = self._ensure_peer_bundle(peer_email)
        return self.validate_peer_bundle(peer_email, bundle)

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
        """Verify and decrypt self-encrypted data if encryption is enabled and keys are available."""
        if self.use_encryption and self.has_my_keys():
            self.verify_message_from_self(data)
            return self.decrypt_for_self(data)
        return data

    # ========== Persistence ==========

    def _pinned_bundles(self) -> dict[str, dict]:
        return {
            peer.email: peer.public_encryption_bundle
            for peer in self._peers
            if peer.public_encryption_bundle is not None
        }

    def _key_file_data(self) -> dict:
        """The key file as this store would write it from memory."""
        keys = self._ensure_private_keys()
        return {
            "version": CRYPTO_KEYS_VERSION,
            "email": self.email,
            "keys_jwk": keys.to_jwks(),
            PEER_BUNDLES_KEY: self._pinned_bundles(),
        }

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

    def _stored_pin(self, peer_email: str) -> dict | None:
        """The pin recorded for ``peer_email`` in the key file, if any."""
        data = self._read_key_file()
        bundles = data.get(PEER_BUNDLES_KEY) if data else None
        pin = bundles.get(peer_email) if isinstance(bundles, dict) else None
        return pin if isinstance(pin, dict) else None

    def _valid_stored_pin(self, peer_email: str) -> dict | None:
        """The stored pin for ``peer_email``, or None when absent or no longer valid."""
        pin = self._stored_pin(peer_email)
        if pin is None:
            return None
        try:
            parse_and_validate_bundle(peer_email, pin)
        except InvalidPeerBundleError as e:
            logger.warning(f"Ignoring stored key pin for {peer_email}: {e}")
            return None
        return pin

    def _persist_peer_bundle(self, peer_email: str, bundle: dict) -> None:
        """Record one pin in the key file, keeping pins other processes wrote.

        The file is re-read and only this peer's entry is replaced, so two
        processes on one datasite (a notebook and syft-bg) do not erase each
        other's pins. The lock covers the read, the change and the write, so
        neither process writes over a pin the other recorded in between.
        """
        if self._keys_path is None or self._private_keys is None:
            return
        with _locked_key_file(self._keys_path):
            data = self._read_key_file() or self._key_file_data()
            _write_key_file(self._keys_path, _with_pin(data, peer_email, bundle))

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
        store = cls(email=data["email"], use_encryption=True)
        store._private_keys = syc.SyftPrivateKeys.from_jwks(data["keys_jwk"])
        store._keys_path = Path(path)
        store._peers = _peers_from_stored_pins(data.get(PEER_BUNDLES_KEY, {}))
        return store

    @classmethod
    def create(
        cls,
        email: str,
        use_encryption: bool = False,
        keys_path: Path | str | None = None,
    ) -> "PeerStore":
        """Build a PeerStore, loading or generating encryption keys when enabled.

        - encryption off: return a plain store (no keys).
        - encryption on: ``keys_path`` is required — the participant's own
          per-datasite key file (``<syftbox_folder>/<email>/private/crypto_keys.json``,
          see :func:`datasite_crypto_keys_path`). Load it when present, else
          generate a fresh key pair and persist it there.

        Scoping keys to a per-datasite file lets several identities run on one
        machine (e.g. two data owners in the same notebook) without colliding, and
        ties key lifetime to the datasite folder.

        Raises:
            ValueError: if encryption is on but ``keys_path`` is missing, or the
                existing key file belongs to a different identity.
        """
        if not use_encryption:
            return cls(email=email, use_encryption=False)
        if keys_path is None:
            raise ValueError("keys_path is required when use_encryption is True")
        path = Path(keys_path)
        if path.exists():
            store = cls.load_keys(path)
            if store.email != email:
                raise ValueError(
                    f"Encryption key file {path} belongs to {store.email!r}, "
                    f"not {email!r}"
                )
            return store
        else:
            # write keys to passed path
            store = cls(email=email, use_encryption=True)
            store.generate_keys()
            store.save_keys(path)
            return store


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


def _with_pin(data: dict, peer_email: str, bundle: dict) -> dict:
    """``data`` from the key file, with ``bundle`` recorded as the pin of ``peer_email``."""
    bundles = data.get(PEER_BUNDLES_KEY)
    if not isinstance(bundles, dict):
        bundles = {}
    bundles[peer_email] = bundle
    data[PEER_BUNDLES_KEY] = bundles
    return data


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


def _peers_from_stored_pins(bundles: dict[str, dict]) -> List[Peer]:
    """Peers carrying the pins in the key file, minus any pin that no longer validates.

    The key file is private and never synced, so a bad pin there is corruption,
    not an attack. It is dropped, and the peer's bundle is read from Drive and
    pinned again on the next load.
    """
    peers = []
    for email, bundle in bundles.items():
        try:
            parse_and_validate_bundle(email, bundle)
        except InvalidPeerBundleError as e:
            logger.warning(f"Dropping stored key pin for {email}: {e}")
            continue
        peers.append(
            Peer(email=email, public_encryption_bundle=bundle, use_encryption=True)
        )
    return peers
