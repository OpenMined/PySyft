"""PeerStore: merged peer list + encryption key management.

Holds peers, encryption keys, and the encryption policy of this client.
Shared between PeerManager and all ConnectionRouter instances.
"""

import json
import logging
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import portalocker
import syft_crypto_python as syc
from pydantic import BaseModel, PrivateAttr, model_validator

from syft.sync.encryption_policy import (
    EncryptionPolicy,
    EncryptionPolicyConflictError,
    PeerEncryptionPolicyUnknownError,
    PeerKeysNotReadyError,
    UnencryptedMessageError,
    pair_conflicts,
    pair_uses_encryption,
    policy_from_use_encryption,
)

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
# Key in the crypto key file that records the policy. Every process that
# loads the file (a notebook and syft-bg) then uses the same policy. A file
# without it was written when only required clients had keys.
ENCRYPTION_POLICY_KEY = "encryption_policy"
# Key in the crypto key file that says the files of this datasite at rest
# (event log, checkpoints, rolling state) are encrypted. A required client
# encrypts them once and sets it: an optional client, or an earlier client
# without encryption, can have written plaintext there.
AT_REST_ENCRYPTED_KEY = "at_rest_encrypted"
# Key in the crypto key file that lists the peers seen to require encryption.
# The version file that advertises a policy is not signed, so such a peer
# stays required here until the user trusts a new key for it.
REQUIRED_PEERS_KEY = "required_peers"
# An earlier client without a bundle for us is read again on the load after
# the first miss, then once every this many peer loads.
LEGACY_BUNDLE_RECHECK_LOADS = 10


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
    encryption_policy: EncryptionPolicy = EncryptionPolicy.OPTIONAL

    _private_keys: syc.SyftPrivateKeys | None = PrivateAttr(default=None)
    _peers: List[Peer] = PrivateAttr(default_factory=list)
    # Where pinned peer bundles persist (the private key file). None when the
    # store was built in memory, e.g. in tests; pins then live only in memory.
    _keys_path: Path | None = PrivateAttr(default=None)
    # Peers whose version file was read again since the last ``set_peers``.
    _policy_rereads: set[str] = PrivateAttr(default_factory=set)
    # Earlier clients known to encrypt. Their version file does not say, so
    # the policy comes from a Drive read; a found bundle is read once.
    _legacy_policies: dict[str, EncryptionPolicy] = PrivateAttr(default_factory=dict)
    # Number of peer loads (``set_peers``) so far.
    _load_count: int = PrivateAttr(default=0)
    # Earlier clients without a found bundle: the load at which none was found.
    _bundle_missing_since: dict[str, int] = PrivateAttr(default_factory=dict)
    # Peers seen to require encryption (see REQUIRED_PEERS_KEY).
    _required_peers: set[str] = PrivateAttr(default_factory=set)
    # Whether the files at rest are known to be encrypted.
    _at_rest_encrypted: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _policy_from_use_encryption(cls, data: Any) -> Any:
        return policy_from_use_encryption(data)

    @property
    def use_encryption(self) -> bool:
        """True when this client requires encryption with every peer and at rest."""
        return self.encryption_policy == EncryptionPolicy.REQUIRED

    @use_encryption.setter
    def use_encryption(self, value: bool) -> None:
        self.encryption_policy = EncryptionPolicy.coerce(value)

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
                pinned peer bundle are missing.
            EncryptionPolicyConflictError: when the policies of the pair conflict.
        """
        if not self.peer_policy_known(email):
            raise PeerEncryptionPolicyUnknownError(email)
        if self.peer_conflicts(email):
            raise EncryptionPolicyConflictError(email)
        if not self.peer_uses_encryption(email):
            return data
        if not self.encryption_ready(email):
            raise PeerKeysNotReadyError(email)
        return self.encrypt(email, data)

    def decrypt_and_verify_if_needed(self, email: str, data: bytes) -> bytes:
        """Verify and decrypt an envelope from ``email``; pass plaintext through.

        When the pair encrypts, plaintext is refused, so that a peer or the
        storage provider cannot downgrade the pair. When the pair does not
        encrypt, an envelope is still accepted: the peer can have started to
        require encryption after the last load.

        Raises:
            UnencryptedMessageError: when the pair encrypts and ``data`` is
                plaintext.
            PeerKeysNotReadyError: when ``data`` is an envelope, but our keys
                or the pinned peer bundle are missing.
            EncryptionPolicyConflictError: when the policies of the pair conflict.
        """
        if self.peer_conflicts(email):
            raise EncryptionPolicyConflictError(email)
        if not self.is_envelope(data):
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
        """The policy of ``email``, or None when not known.

        A peer seen to require encryption stays required, whatever its
        version file says now (see ``remember_required_peer``).
        """
        peer = self.get_cached_peer(email)
        if peer is None:
            return None
        if email in self._required_peers:
            return EncryptionPolicy.REQUIRED
        return peer.encryption_policy

    def is_remembered_required(self, email: str) -> bool:
        return email in self._required_peers

    def remember_required_peer(self, email: str) -> None:
        """Keep ``email`` required, in memory and in the key file."""
        if email not in self._required_peers:
            self._required_peers.add(email)
            self._persist_required_peers()

    def forget_required_peer(self, email: str) -> None:
        """Stop keeping ``email`` required. For a trusted key change only."""
        if email in self._required_peers:
            self._required_peers.discard(email)
            self._persist_required_peers(removed=email)

    def _persist_required_peers(self, removed: str | None = None) -> None:
        """Record the remembered peers in the key file, with those of other processes."""
        if self._keys_path is None or self._private_keys is None:
            return
        with _locked_key_file(self._keys_path):
            data = self._read_key_file() or self._key_file_data()
            stored = set(data.get(REQUIRED_PEERS_KEY, [])) - {removed}
            self._required_peers |= stored
            data = {**data, REQUIRED_PEERS_KEY: sorted(self._required_peers)}
            _write_key_file(self._keys_path, data)

    def peer_policy_known(self, email: str) -> bool:
        """Whether this client can select the format for ``email``.

        A required client encrypts with every peer, so it needs no peer policy.
        """
        return self.use_encryption or self.peer_policy(email) is not None

    def encryption_ready(self, email: str) -> bool:
        """Whether our keys and the pinned bundle of ``email`` exist."""
        return self.has_my_keys() and self.has_peer_bundle(email)

    def peer_conflicts(self, email: str) -> bool:
        """Whether this client has encryption off and ``email`` requires it, or the reverse."""
        return pair_conflicts(self.encryption_policy, self.peer_policy(email))

    def peer_uses_encryption(self, email: str) -> bool:
        """Whether the pair with ``email`` encrypts. False when the peer policy is not known."""
        return pair_uses_encryption(self.encryption_policy, self.peer_policy(email))

    def set_peer(self, peer: Peer) -> None:
        self._keep_pinned_bundle(peer)
        for i, p in enumerate(self._peers):
            if p.email == peer.email:
                self._peers[i] = peer
                return
        self._peers.append(peer)

    def add_peer(self, peer: Peer) -> None:
        self._keep_pinned_bundle(peer)
        self._peers.append(peer)

    def set_peers(self, peers: List[Peer]) -> None:
        for p in peers:
            self._keep_pinned_bundle(p)
        self._peers = peers
        self._policy_rereads.clear()
        self._load_count += 1

    @property
    def peers_loaded(self) -> bool:
        """Whether a peer load ran. Before it, no recipient is known to be a peer."""
        return self._load_count > 0

    def legacy_policy(
        self, email: str, published_bundle: Callable[[], bool]
    ) -> Optional[EncryptionPolicy]:
        """The policy of the earlier client ``email``, or None while not known.

        Such a client publishes a bundle for us only when it encrypts. One
        that does not has no keys and cannot decrypt, so it is off. A bundle
        can be not yet visible on Drive, so a client without one is unknown
        during the peer load that first misses it, and off from the next.
        A found bundle is kept for the process.
        """
        if email in self._legacy_policies:
            return self._legacy_policies[email]
        if published_bundle():
            self._legacy_policies[email] = EncryptionPolicy.REQUIRED
            self._bundle_missing_since.pop(email, None)
            return EncryptionPolicy.REQUIRED
        first_missed = self._bundle_missing_since.setdefault(email, self._load_count)
        return EncryptionPolicy.OFF if self._load_count > first_missed else None

    def legacy_bundle_needs_read(self, email: str) -> bool:
        """Whether to read again if the earlier client ``email`` published a bundle.

        Not once one was found. After a miss: on the next load, then once
        every LEGACY_BUNDLE_RECHECK_LOADS loads.
        """
        if email in self._legacy_policies:
            return False
        first_missed = self._bundle_missing_since.get(email)
        if first_missed is None:
            return True
        loads = self._load_count - first_missed
        return loads <= 1 or loads % LEGACY_BUNDLE_RECHECK_LOADS == 0

    def take_policy_reread(self, email: str) -> bool:
        """True once per peer load: the caller may read the version file again."""
        if email in self._policy_rereads:
            return False
        self._policy_rereads.add(email)
        return True

    def _keep_pinned_bundle(self, incoming: Peer) -> None:
        """Make ``incoming`` carry the pinned bundle for its email, if any.

        Peers loaded from ``SYFT_peers.json`` carry whatever bundle the Drive
        copy holds. A pin always wins over that copy: a differing Drive bundle
        is reported and dropped. Without a pin, a valid Drive bundle becomes the
        pin (trust on first use) and an invalid one is dropped.

        An optional client without keys has no key file to pin in. It keeps
        only a bundle it set itself (``set_peer_bundle``) and drops the Drive
        copy; ``load_peers`` pins the published bundle once it has keys.
        """
        if not self.use_encryption and not self.has_my_keys():
            cached = self.get_cached_peer(incoming.email)
            incoming.public_encryption_bundle = (
                cached.public_encryption_bundle if cached is not None else None
            )
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
        unsaved_pins: list[tuple[str, dict]] = []
        with _locked_key_file(self._keys_path):
            if self._keys_path.exists():
                unsaved_pins = self._take_key_file(self.load_keys(self._keys_path))
            else:
                self.generate_keys()
                _write_key_file(self._keys_path, self._key_file_data())
        # Saving a pin takes the lock again, so it waits until here.
        for email, bundle in unsaved_pins:
            self._persist_peer_bundle(email, bundle)
        self._persist_required_peers()

    def _adopt_key_file(self, loaded: "PeerStore") -> None:
        """Take the keys, pins and recorded policy of ``loaded``, a store read from our key file."""
        for email, bundle in self._take_key_file(loaded):
            self._persist_peer_bundle(email, bundle)
        self._persist_required_peers()

    def _take_key_file(self, loaded: "PeerStore") -> list[tuple[str, dict]]:
        """Like ``_adopt_key_file``, but return the pins to save, without file I/O.

        These are pins set in memory before there were keys, e.g. from
        attestation, that the key file does not hold yet.
        """
        if loaded.email != self.email:
            raise ValueError(
                f"Encryption key file {self._keys_path} belongs to "
                f"{loaded.email!r}, not {self.email!r}"
            )
        self._private_keys = loaded._private_keys
        self._at_rest_encrypted = loaded._at_rest_encrypted
        self._required_peers |= loaded._required_peers
        if loaded.use_encryption:
            self.encryption_policy = EncryptionPolicy.REQUIRED
        if not self._peers:
            self._peers = loaded._peers
            return []
        stored = {p.email: p.public_encryption_bundle for p in loaded._peers}
        unsaved_pins = []
        for peer in self._peers:
            if peer.email in stored:
                # The pin in the key file wins.
                peer.public_encryption_bundle = stored[peer.email]
            elif peer.public_encryption_bundle is not None:
                unsaved_pins.append((peer.email, peer.public_encryption_bundle))
        return unsaved_pins

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
            if old_fp != new_fp:
                # A peer drops required only with a new key file, so a new key.
                self.forget_required_peer(peer_email)
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
            ENCRYPTION_POLICY_KEY: self.encryption_policy.value,
            AT_REST_ENCRYPTED_KEY: self._at_rest_encrypted,
            REQUIRED_PEERS_KEY: sorted(self._required_peers),
        }

    def reload_policy_from_key_file(self) -> None:
        """Take a required policy that another process recorded in the key file.

        A notebook that started optional must not write plaintext at rest once
        syft-bg made the datasite required.

        Raises:
            ValueError: when this client has encryption off.
        """
        if self.use_encryption or self._keys_path is None:
            return
        # Most syncs find no change: read the policy before the keys.
        data = self._read_key_file()
        recorded = (
            data.get(ENCRYPTION_POLICY_KEY, EncryptionPolicy.REQUIRED.value)
            if data
            else None
        )
        if recorded != EncryptionPolicy.REQUIRED.value:
            return
        loaded = self.load_keys(self._keys_path)
        if self.encryption_policy == EncryptionPolicy.OFF:
            raise _off_refused_error(self._keys_path)
        self._adopt_key_file(loaded)

    def needs_at_rest_encryption(self) -> bool:
        """Whether this client is required, but its files at rest can be plaintext."""
        return self.use_encryption and not self._at_rest_encrypted

    def encrypt_files_at_rest_once(self, encrypt_files: Callable[[], None]) -> None:
        """Run ``encrypt_files`` unless a process already did, then record it.

        A lock of its own covers the check, the run and the record, so two
        processes on one datasite do not both encrypt the files. The key file
        lock is taken only to record, so the run does not block key work.
        """
        self.ensure_keys()
        if self._keys_path is None:
            encrypt_files()
        else:
            at_rest_lock = self._keys_path.with_name(self._keys_path.name + ".at_rest")
            with _locked_key_file(at_rest_lock):
                data = self._read_key_file() or {}
                if not data.get(AT_REST_ENCRYPTED_KEY):
                    encrypt_files()
                    with _locked_key_file(self._keys_path):
                        data = self._read_key_file() or self._key_file_data()
                        data = {**data, AT_REST_ENCRYPTED_KEY: True}
                        _write_key_file(self._keys_path, data)
        self._at_rest_encrypted = True

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
        policy = data.get(ENCRYPTION_POLICY_KEY, EncryptionPolicy.REQUIRED.value)
        store = cls(email=data["email"], encryption_policy=EncryptionPolicy(policy))
        store._at_rest_encrypted = bool(data.get(AT_REST_ENCRYPTED_KEY))
        store._required_peers = set(data.get(REQUIRED_PEERS_KEY, []))
        store._private_keys = syc.SyftPrivateKeys.from_jwks(data["keys_jwk"])
        store._keys_path = Path(path)
        store._peers = _peers_from_stored_pins(data.get(PEER_BUNDLES_KEY, {}))
        return store

    @classmethod
    def create(
        cls,
        email: str,
        encryption_policy: EncryptionPolicy = EncryptionPolicy.OPTIONAL,
        keys_path: Path | str | None = None,
        use_encryption: bool | None = None,
    ) -> "PeerStore":
        """Build a PeerStore for ``email`` with its per-datasite key file.

        ``keys_path`` is the participant's own key file
        (``<syftbox_folder>/<email>/private/crypto_keys.json``, see
        :func:`datasite_crypto_keys_path`). When the file exists, the store
        loads its keys and pins. A required policy in the file wins over an
        optional ``encryption_policy``, so all processes on a datasite agree.

        Without a file, a required store generates keys and saves them. An
        optional store has no keys until :meth:`ensure_keys`.

        ``use_encryption`` is the earlier bool form of ``encryption_policy``.

        Scoping keys to a per-datasite file lets several identities run on one
        machine (e.g. two data owners in the same notebook) without colliding, and
        ties key lifetime to the datasite folder.

        Raises:
            ValueError: if the policy is required but ``keys_path`` is missing,
                or the existing key file belongs to a different identity.
        """
        if use_encryption is not None:
            encryption_policy = EncryptionPolicy.coerce(use_encryption)
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
            store._adopt_key_file(loaded)
            store._record_required_policy()
        elif store.use_encryption:
            store.generate_keys()
            store.save_keys(store._keys_path)
            _warn_key_file_holds_data(store._keys_path)
        return store


def _off_refused_error(keys_path: Path) -> ValueError:
    return ValueError(
        f"The key file {keys_path} requires encryption: the files of this "
        "datasite at rest are encrypted. Encryption cannot be off for it. Use a "
        "new SyftBox folder for a client with encryption off."
    )


def _warn_key_file_holds_data(keys_path: Path) -> None:
    """Warn once, when a datasite starts to require encryption."""
    warnings.warn(
        "This datasite now requires encryption, and every later session keeps "
        "that setting. Its files at rest are encrypted with the key in "
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
        peers.append(Peer(email=email, public_encryption_bundle=bundle))
    return peers
