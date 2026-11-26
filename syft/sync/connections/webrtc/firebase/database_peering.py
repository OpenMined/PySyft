"""Peering through the Realtime Database: peer requests, acl/, key bundles and
peer info.

infra/firebase/database.rules.json has the paths and who may read and write them.
The peer list stays on local disk.
"""

from dataclasses import dataclass
from pathlib import Path

from syft_migration import MigrationError

from syft.sync.connections.naming import PEERS_META_KEY, SYFT_PEERS_VERSION
from syft.sync.connections.protocols import PeeringConnection
from syft.sync.connections.webrtc.firebase.database import (
    RealtimeDatabase,
    email_key,
    email_of_key,
)
from syft.sync.peers.peer import PeerState
from syft.sync.sync.caches.persisted_dict import PersistedDict
from syft.sync.sync.constants import CACHE_DIR
from syft.sync.version.exceptions import UnreadablePeerInfoError
from syft.sync.version.peer_info import PeerInfo

# The peer list, under the syftbox folder.
PEER_REGISTRY_PATH = Path(CACHE_DIR) / "webrtc" / "SYFT_peers.json"
# A request from a peer in one of these states needs no decision.
DECIDED_STATES = (PeerState.ACCEPTED.value, PeerState.REJECTED.value)


def requests_path(receiver: str) -> str:
    """Return the node of the requests to peer with ``receiver``."""
    return f"requests/{email_key(receiver)}"


def acl_path(owner: str) -> str:
    """Return the node of the peers that ``owner`` accepts."""
    return f"acl/{email_key(owner)}"


def key_bundle_path(email: str) -> str:
    return f"keys/{email_key(email)}"


def peer_info_path(owner: str, reader: str) -> str:
    """Return the node of the peer info that ``owner`` shares with ``reader``."""
    return f"info/{email_key(owner)}/{email_key(reader)}"


@dataclass(frozen=True)
class DatabasePeeringConfig:
    # The email of this client. Peers see it as spelled here.
    email: str
    syftbox_folder: Path

    def __post_init__(self) -> None:
        if not self.email:
            raise ValueError("email must not be empty")


class DatabasePeeringConnection(PeeringConnection):
    """``PeeringConnection`` through the Realtime Database. It works from any
    thread."""

    def __init__(self, config: DatabasePeeringConfig, database: RealtimeDatabase):
        self._config = config
        self._database = database
        self._registry = PersistedDict(path=config.syftbox_folder / PEER_REGISTRY_PATH)

    def whoami(self) -> str:
        return self._config.email

    def open_channels(self, peer: str) -> None:
        """Ask ``peer`` to peer, and let it send signals to this client.

        WebRTC opens the channel when both peers are online.
        """
        me = self._config.email
        self._database.update(
            {
                f"{requests_path(peer)}/{email_key(me)}": me,
                f"{acl_path(me)}/{email_key(peer)}": True,
            }
        )

    def incoming_requests(self) -> list[str]:
        """Return the emails that asked to peer, and have no decision yet.

        An email that is in the peer list comes back as the peer list spells it.
        """
        requests = self._database.get(requests_path(self._config.email))
        senders = [] if requests is None else list(requests.values())
        registry = dict(self._registry.items())
        # email in lower case -> the email as the peer list spells it
        spellings = {email.lower(): email for email in registry}
        emails = [spellings.get(sender.lower(), sender) for sender in senders]
        return [
            email
            for email in emails
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
        """Publish the key bundle of this client, the same for every peer."""
        self._database.put(key_bundle_path(self._config.email), bundle_json)

    def grant_key_bundle_read(self, peer: str) -> None:
        """Do nothing: every signed-in user can read keys/, the key directory."""

    def read_own_key_bundle(self, peer: str) -> str | None:
        return self._database.get(key_bundle_path(self._config.email))

    def read_peer_key_bundle(self, peer: str) -> str | None:
        return self._database.get(key_bundle_path(peer))

    def publish_peer_info(self, peer_info: PeerInfo) -> None:
        """Write ``peer_info`` for this client, and for each peer it accepts."""
        me = self._config.email
        accepted = self._database.get(acl_path(me))
        readers = (
            [] if accepted is None else [key for key, ok in accepted.items() if ok]
        )
        paths = [peer_info_path(me, me)] + [
            peer_info_path(me, email_of_key(reader)) for reader in readers
        ]
        self._write_peer_info(paths, peer_info)

    def read_own_peer_info(self) -> PeerInfo | None:
        return self._read_peer_info(self._config.email)

    def read_peer_info(self, peer: str) -> PeerInfo | None:
        return self._read_peer_info(peer)

    def grant_peer_info_read(self, peer: str, peer_info: PeerInfo) -> None:
        """Write ``peer_info`` for this client and for ``peer``."""
        me = self._config.email
        self._write_peer_info(
            [peer_info_path(me, me), peer_info_path(me, peer)], peer_info
        )

    def _write_peer_info(self, paths: list[str], peer_info: PeerInfo) -> None:
        data = peer_info.to_json()
        self._database.update({path: data for path in paths})

    def _read_peer_info(self, owner: str) -> PeerInfo | None:
        """Return the peer info that ``owner`` shares with this client, or None
        if it shares none."""
        data = self._database.get(peer_info_path(owner, self._config.email))
        if data is None:
            return None
        try:
            return PeerInfo.from_json(data)
        except (ValueError, MigrationError) as err:
            raise UnreadablePeerInfoError(
                f"Could not read the peer info of {owner}: {err}"
            ) from err
