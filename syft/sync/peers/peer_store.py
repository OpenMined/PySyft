"""PeerStore: merged peer list + encryption key management.

Holds peers, encryption keys, and a use_encryption flag.
Shared between PeerManager and all ConnectionRouter instances.
"""

import json
from pathlib import Path
from typing import List, Optional

import syft_crypto_python as syc
from pydantic import BaseModel, PrivateAttr

from syft.sync.peers.peer import Peer

# Encryption key bundles persist inside the participant's own SyftBox datasite
# folder, under private/ (which is never synced to Drive). This scopes keys per
# identity by location, so several identities can run on one machine without
# colliding, and delete_syftbox removes them together with the folder.
PRIVATE_DIR_NAME = "private"
CRYPTO_KEYS_FILENAME = "crypto_keys.json"


def datasite_crypto_keys_path(syftbox_folder: Path | str, email: str) -> Path:
    """Per-datasite key file: ``<syftbox_folder>/<email>/private/crypto_keys.json``."""
    return Path(syftbox_folder) / email / PRIVATE_DIR_NAME / CRYPTO_KEYS_FILENAME


class PeerStore(BaseModel):
    """Manages peers and encryption keys for E2E encryption."""

    model_config = {"arbitrary_types_allowed": True}

    email: str
    use_encryption: bool = False

    _private_keys: syc.SyftPrivateKeys | None = PrivateAttr(default=None)
    _peers: List[Peer] = PrivateAttr(default_factory=list)

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
        for i, p in enumerate(self._peers):
            if p.email == peer.email:
                self._peers[i] = peer
                return
        self._peers.append(peer)

    def add_peer(self, peer: Peer) -> None:
        peer.use_encryption = self.use_encryption
        self._peers.append(peer)

    def set_peers(self, peers: List[Peer]) -> None:
        for p in peers:
            p.use_encryption = self.use_encryption
        self._peers = peers

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
        """Verify and decrypt self-encrypted data if encryption is enabled and keys are available."""
        if self.use_encryption and self.has_my_keys():
            self.verify_message_from_self(data)
            return self.decrypt_for_self(data)
        return data

    # ========== Persistence ==========

    def save_keys(self, path: Path) -> None:
        keys = self._ensure_private_keys()
        data = {
            "email": self.email,
            "keys_jwk": keys.to_jwks(),
            "peer_bundles": {
                peer.email: peer.public_encryption_bundle
                for peer in self._peers
                if peer.public_encryption_bundle is not None
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_keys(cls, path: Path) -> "PeerStore":
        data = json.loads(Path(path).read_text())
        store = cls(email=data["email"], use_encryption=True)
        store._private_keys = syc.SyftPrivateKeys.from_jwks(data["keys_jwk"])
        for email, bundle_dict in data.get("peer_bundles", {}).items():
            peer = Peer(
                email=email,
                public_encryption_bundle=bundle_dict,
                use_encryption=True,
            )
            store._peers.append(peer)
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
