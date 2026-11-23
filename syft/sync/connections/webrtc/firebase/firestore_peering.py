"""Peering through Firestore: peer requests, key bundles and peer info.

firebase/firestore.rules has the paths and who may read and write them. The
peer list stays on local disk.
"""

from dataclasses import dataclass
from pathlib import Path

from syft_migration import MigrationError

from syft.sync.connections.naming import PEERS_META_KEY, SYFT_PEERS_VERSION
from syft.sync.connections.protocols import PeeringConnection
from syft.sync.connections.webrtc.firebase.firestore import FirestoreClient, SetFields
from syft.sync.peers.peer import PeerState
from syft.sync.sync.caches.persisted_dict import PersistedDict
from syft.sync.sync.constants import CACHE_DIR
from syft.sync.version.exceptions import UnreadablePeerInfoError
from syft.sync.version.peer_info import PeerInfo

# The peer list, under the syftbox folder.
PEER_REGISTRY_PATH = Path(CACHE_DIR) / "webrtc" / "SYFT_peers.json"
# A request from a peer in one of these states needs no decision.
DECIDED_STATES = (PeerState.ACCEPTED.value, PeerState.REJECTED.value)
REQUEST_EMAIL_FIELD = "email"
KEY_BUNDLE_FIELD = "key_bundle"
PEER_INFO_FIELD = "peer_info"


def peer_requests_path(receiver: str) -> str:
    """Return the collection of the requests to peer with ``receiver``."""
    return f"peer_requests/{receiver.lower()}/from"


def shared_with_path(owner: str) -> str:
    """Return the collection of what ``owner`` shares with each reader."""
    return f"shared/{owner.lower()}/with"


def shared_path(owner: str, reader: str) -> str:
    """Return the document of what ``owner`` shares with ``reader``."""
    return f"{shared_with_path(owner)}/{reader.lower()}"


@dataclass(frozen=True)
class FirestorePeeringConfig:
    # The email of this client. Peers see it as spelled here.
    email: str
    syftbox_folder: Path

    def __post_init__(self) -> None:
        if not self.email:
            raise ValueError("email must not be empty")


class FirestorePeeringConnection(PeeringConnection):
    """``PeeringConnection`` through Firestore. It works from any thread.

    Firestore has no grant on one document. Therefore what this client shares
    with a peer goes into a document that only the two of them read.
    """

    def __init__(self, config: FirestorePeeringConfig, firestore: FirestoreClient):
        self._config = config
        self._firestore = firestore
        self._registry = PersistedDict(path=config.syftbox_folder / PEER_REGISTRY_PATH)

    def whoami(self) -> str:
        return self._config.email

    def open_channels(self, peer: str) -> None:
        """Ask ``peer`` to peer.

        WebRTC opens the channel when both peers are online.
        """
        request = SetFields(
            f"{peer_requests_path(peer)}/{self._config.email.lower()}",
            {REQUEST_EMAIL_FIELD: self._config.email},
        )
        self._firestore.write_documents([request])

    def incoming_requests(self) -> list[str]:
        """Return the emails that asked to peer, and have no decision yet.

        An email that is in the peer list comes back as the peer list spells it.
        """
        read = self._firestore.query_collection(peer_requests_path(self._config.email))
        registry = dict(self._registry.items())
        # email in lower case -> the email as the peer list spells it
        spellings = {email.lower(): email for email in registry}
        senders = [request.text(REQUEST_EMAIL_FIELD) for request in read.documents]
        requests = [spellings.get(email.lower(), email) for email in senders]
        return [
            email
            for email in requests
            if email not in registry or registry[email]["state"] not in DECIDED_STATES
        ]

    def read_peer_registry(self, force_download: bool = False) -> dict:
        """Return the peer list on local disk.

        Each call reads the file, so ``force_download`` changes nothing.
        """
        return dict(self._registry.items())

    def update_peer_registry(
        self, peer: str, state: str, public_encryption_bundle: dict | None = None
    ) -> None:
        def update(entry: dict | None) -> dict:
            new_entry = {} if entry is None else dict(entry)
            new_entry["state"] = state
            if public_encryption_bundle is not None:
                new_entry["public_encryption_bundle"] = public_encryption_bundle
            return new_entry

        self._registry.update_entry(peer, update)
        self._registry.set(PEERS_META_KEY, {"version": SYFT_PEERS_VERSION})

    def publish_key_bundle(self, peer: str, bundle_json: str) -> None:
        path = shared_path(self._config.email, peer)
        self._firestore.write_documents(
            [SetFields(path, {KEY_BUNDLE_FIELD: bundle_json})]
        )

    def grant_key_bundle_read(self, peer: str) -> None:
        """Do nothing: ``publish_key_bundle`` writes the key bundle into a
        document that ``peer`` reads."""

    def read_own_key_bundle(self, peer: str) -> str | None:
        return self._read_shared_field(self._config.email, peer, KEY_BUNDLE_FIELD)

    def read_peer_key_bundle(self, peer: str) -> str | None:
        return self._read_shared_field(peer, self._config.email, KEY_BUNDLE_FIELD)

    def publish_peer_info(self, peer_info: PeerInfo) -> None:
        """Write ``peer_info`` for this client, and for each peer that can read
        the peer info of this client."""
        read = self._firestore.query_collection(shared_with_path(self._config.email))
        readers = [
            shared.path for shared in read.documents if PEER_INFO_FIELD in shared.fields
        ]
        own = shared_path(self._config.email, self._config.email)
        self._write_peer_info(sorted({own, *readers}), peer_info)

    def read_own_peer_info(self) -> PeerInfo | None:
        return self._read_peer_info(self._config.email)

    def read_peer_info(self, peer: str) -> PeerInfo | None:
        return self._read_peer_info(peer)

    def grant_peer_info_read(self, peer: str, peer_info: PeerInfo) -> None:
        """Write ``peer_info`` for this client and for ``peer``."""
        paths = [
            shared_path(self._config.email, self._config.email),
            shared_path(self._config.email, peer),
        ]
        self._write_peer_info(paths, peer_info)

    def _write_peer_info(self, paths: list[str], peer_info: PeerInfo) -> None:
        data = peer_info.to_json()
        self._firestore.write_documents(
            [SetFields(path, {PEER_INFO_FIELD: data}) for path in paths]
        )

    def _read_peer_info(self, owner: str) -> PeerInfo | None:
        data = self._read_shared_field(owner, self._config.email, PEER_INFO_FIELD)
        if data is None:
            return None
        try:
            return PeerInfo.from_json(data)
        except (ValueError, MigrationError) as err:
            raise UnreadablePeerInfoError(
                f"Could not read the peer info of {owner}: {err}"
            ) from err

    def _read_shared_field(self, owner: str, reader: str, field: str) -> str | None:
        """Return ``field`` of what ``owner`` shares with ``reader``, or None if
        ``owner`` did not share it."""
        shared = self._firestore.get_document(shared_path(owner, reader))
        if shared is None or field not in shared.fields:
            return None
        return shared.text(field)
