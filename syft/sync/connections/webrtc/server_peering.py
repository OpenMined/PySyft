"""Peering through the syft signaling server.

Peer requests, key bundles and peer info go through the server. The peer list
stays on local disk.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from syft_migration import MigrationError
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.protocols import PeeringConnection
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.peers.peer import PeerState
from syft.sync.sync.caches.persisted_dict import PersistedDict
from syft.sync.sync.constants import CACHE_DIR
from syft.sync.version.peer_info import PeerInfo

# The peer list, under the syftbox folder.
PEER_REGISTRY_PATH = Path(CACHE_DIR) / "webrtc" / "SYFT_peers.json"
REQUEST_TIMEOUT_SECONDS = 20
PEER_INFO_DOCUMENT = "peer_info"
# A request from a peer in one of these states needs no decision.
DECIDED_STATES = (PeerState.ACCEPTED.value, PeerState.REJECTED.value)


def key_bundle_document(reader: str) -> str:
    """Return the name of the key bundle document for ``reader``."""
    return f"key_bundle_for:{reader.lower()}"


@dataclass(frozen=True)
class ServerPeeringConfig:
    server: WebRTCConfig
    # The email of this client. The server trusts it, and shows it to peers as
    # spelled here.
    email: str
    syftbox_folder: Path

    def __post_init__(self) -> None:
        if not self.email:
            raise ValueError("email must not be empty")


class ServerPeeringConnection(PeeringConnection):
    """``PeeringConnection`` through the signaling server.

    Each call opens a WebSocket for one request, so the connection works from
    any thread.
    """

    def __init__(self, config: ServerPeeringConfig):
        self._config = config
        self._registry = PersistedDict(path=config.syftbox_folder / PEER_REGISTRY_PATH)

    def whoami(self) -> str:
        return self._config.email

    def open_channels(self, peer: str) -> None:
        """Ask ``peer`` to peer.

        WebRTC opens the channel when both peers are online.
        """
        self._send_request({"type": "request_peer", "peer": peer})

    def incoming_requests(self) -> list[str]:
        """Return the emails that asked to peer, and have no decision yet.

        An email that is in the peer list comes back as the peer list spells it.
        """
        reply = self._send_request({"type": "list_peer_requests"})
        registry = dict(self._registry.items())
        # email in lower case -> the email as the peer list spells it
        spellings = {email.lower(): email for email in registry}
        requests = [spellings.get(email.lower(), email) for email in reply["emails"]]
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

    def publish_key_bundle(self, peer: str, bundle_json: str) -> None:
        self._put_document(key_bundle_document(peer), bundle_json)

    def grant_key_bundle_read(self, peer: str) -> None:
        self._grant_document(key_bundle_document(peer), peer)

    def read_own_key_bundle(self, peer: str) -> str | None:
        return self._get_document(self._config.email, key_bundle_document(peer))

    def read_peer_key_bundle(self, peer: str) -> str | None:
        return self._get_document(peer, key_bundle_document(self._config.email))

    def publish_peer_info(self, peer_info: PeerInfo) -> None:
        self._put_document(PEER_INFO_DOCUMENT, peer_info.to_json())

    def read_own_peer_info(self) -> PeerInfo | None:
        return self._read_peer_info(self._config.email)

    def read_peer_info(self, peer: str) -> PeerInfo | None:
        return self._read_peer_info(peer)

    def grant_peer_info_read(self, peer: str, peer_info: PeerInfo) -> None:
        if self._get_document(self._config.email, PEER_INFO_DOCUMENT) is None:
            self.publish_peer_info(peer_info)
        self._grant_document(PEER_INFO_DOCUMENT, peer)

    def _read_peer_info(self, owner: str) -> PeerInfo | None:
        data = self._get_document(owner, PEER_INFO_DOCUMENT)
        if data is None:
            return None
        try:
            return PeerInfo.from_json(data)
        except (ValueError, MigrationError) as err:
            raise ValueError(f"Could not read the peer info of {owner}: {err}") from err

    def _put_document(self, name: str, data: str) -> None:
        self._send_request({"type": "put_document", "name": name, "data": data})

    def _grant_document(self, name: str, reader: str) -> None:
        self._send_request({"type": "grant_document", "name": name, "reader": reader})

    def _get_document(self, owner: str, name: str) -> str | None:
        reply = self._send_request(
            {"type": "get_document", "owner": owner, "name": name}
        )
        return reply["data"]

    def _send_request(self, request: dict) -> dict:
        """Send one peering request to the server, and return the answer.

        Raises:
            ConnectionUnavailableError: The server cannot be reached.
            ConnectionError: The server refuses the request.
        """
        credentials = {
            "email": self._config.email,
            "access_key": self._config.server.access_key,
        }
        url = self._config.server.signaling_url
        try:
            with connect(url, open_timeout=REQUEST_TIMEOUT_SECONDS) as connection:
                connection.send(json.dumps({**request, **credentials}))
                reply = json.loads(connection.recv(timeout=REQUEST_TIMEOUT_SECONDS))
        except (OSError, WebSocketException) as err:
            raise ConnectionUnavailableError(
                f"Could not reach the signaling server at {url}: {err}"
            ) from err
        if reply["type"] == "error":
            raise ConnectionError(
                f"The signaling server refused {request['type']}: {reply['reason']}"
            )
        return reply
