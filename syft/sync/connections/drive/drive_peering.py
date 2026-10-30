"""Finding and trusting peers on Google Drive.

The peer list is /SyftBox/SYFT_peers.json, the version document is
/SyftBox/SYFT_version.json, and the key bundles sit in a folder of their own.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel
from syft_migration import MigrationError

from syft.sync.connections.drive.gdrive_client import GOOGLE_FOLDER_MIME_TYPE
from syft.sync.connections.drive.gdrive_layout import (
    GDRIVE_P2P_FOLDER_DATASITE_PREFIX,
    DriveLayout,
    GdriveP2PFolder,
)
from syft.sync.connections.drive.gdrive_retry import (
    execute_with_retries,
    raise_unavailable,
)
from syft.sync.transport.naming import PEERS_META_KEY, SYFT_PEERS_VERSION
from syft.sync.transport.protocols import Peering

if TYPE_CHECKING:
    from syft.sync.version.peer_info import PeerInfo


SYFT_PEERS_FILE = "SYFT_peers.json"

SYFT_VERSION_FILE = "SYFT_version.json"


class GdriveEncryptionBundlesFolder(BaseModel):
    """Folder for encryption bundles: syft_encryption_bundles#email"""

    email: str

    def as_string(self) -> str:
        return f"syft_encryption_bundles#{self.email}"


class DrivePeering(Peering):
    """Identity from the Google account; channels, the peer list, key bundles
    and version files in /SyftBox."""

    def __init__(self, layout: DriveLayout):
        self._layout = layout
        # Encryption bundles folder cache
        self._encryption_bundles_folder_id: str | None = None
        # SYFT_peers.json as loaded: peer email -> {"state", "public_encryption_bundle"},
        # plus "_meta" -> {"version"}. None until the first read.
        self._peers_json_cache: dict[str, dict[str, Any]] | None = None

    def clone_for_thread(self) -> "DrivePeering":
        """Peering over a new Drive client, for use from another thread, with
        these caches."""
        clone = DrivePeering(self._layout.clone_for_thread())
        clone._encryption_bundles_folder_id = self._encryption_bundles_folder_id
        clone._peers_json_cache = (
            dict(self._peers_json_cache) if self._peers_json_cache is not None else None
        )
        return clone

    def reset_caches(self) -> None:
        self._layout.reset_caches()
        self._encryption_bundles_folder_id = None
        self._peers_json_cache = None

    def whoami(self) -> str:
        """Return the email of the Google account behind the Drive service."""
        about = execute_with_retries(
            self._layout.client.drive_service.about().get(fields="user(emailAddress)")
        )
        return about["user"]["emailAddress"]

    def open_channels(self, peer: str):
        """Create inbox + outbox folders for peer's datasite on own drive, share with peer.

        Creates:
        - syft_datasite_{peer}_inbox_{self} — I submit proposals to peer here
        - syft_datasite_{peer}_outbox_{self} — peer pushes events to me here
        """
        # Inbox: I write proposals, peer reads
        inbox_id = self._layout.get_peer_datasite_inbox_id(peer)
        if inbox_id is None:
            inbox_id = self._layout.create_peer_datasite_folder(peer, "inbox")
        self._layout.client.add_permission(inbox_id, peer, write=True)

        # Outbox: peer writes events, I read
        outbox_id = self._layout.get_peer_datasite_outbox_id(peer)
        if outbox_id is None:
            outbox_id = self._layout.create_peer_datasite_folder(peer, "outbox")
        self._layout.client.add_permission(outbox_id, peer, write=True)

    def _get_peers_file_id(self) -> str | None:
        """Find SYFT_peers.json file in /SyftBox folder"""
        syftbox_folder_id = self._layout.get_syftbox_folder_id()
        query = f"name='{SYFT_PEERS_FILE}' and '{syftbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self._layout.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def _download_peers_json(self) -> dict[str, dict[str, Any]]:
        """Fetch peers JSON from GDrive. Returns empty dict if not found.

        Maps each peer email to ``{"state": str, "public_encryption_bundle":
        dict}``, and ``_meta`` to ``{"version": int}``.

        Raises:
            TransportUnavailableError: when Drive cannot be reached.
            ValueError: when the file is not valid JSON.
        """
        with raise_unavailable("Could not read the peers file from Drive."):
            file_id = self._get_peers_file_id()
            if file_id is None:
                return {}
            file_data = self._layout.client.download_file(file_id)
        try:
            return json.loads(file_data.decode("utf-8"))
        except ValueError as e:
            raise ValueError(f"Could not read the peers file: {e}") from e

    def read_peer_registry(
        self, force_download: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Return peers JSON, using the in-memory cache when available."""
        if self._peers_json_cache is not None and not force_download:
            return self._peers_json_cache
        self._peers_json_cache = self._download_peers_json()
        return self._peers_json_cache

    def _write_peers_json(self, peers_data: dict[str, dict[str, Any]]):
        """Write peers JSON to GDrive. Creates or updates the file."""
        peers_data = {
            **peers_data,
            PEERS_META_KEY: {"version": SYFT_PEERS_VERSION},
        }
        syftbox_folder_id = self._layout.get_syftbox_folder_id()
        file_id = self._get_peers_file_id()

        # Convert to JSON bytes
        json_data = json.dumps(peers_data, indent=2)
        file_payload, _ = self._layout.client.create_file_payload(json_data)

        if file_id is None:
            # Create new file
            file_metadata = {
                "name": SYFT_PEERS_FILE,
                "parents": [syftbox_folder_id],
            }
            result = self._layout.client.create_syft_file(file_metadata, file_payload)
            self._peers_json_cache = peers_data
            return result.get("id")
        else:
            # Update existing file
            execute_with_retries(
                self._layout.client.drive_service.files().update(
                    fileId=file_id, media_body=file_payload
                )
            )
            self._peers_json_cache = peers_data
            return file_id

    def update_peer_registry(
        self,
        peer: str,
        state: str,
        public_encryption_bundle: dict | None = None,
    ):
        """Update a single peer's state in the JSON file.

        Preserves existing fields when updating state.
        """
        peers_data = self.read_peer_registry()
        existing = peers_data.get(peer, {})
        existing["state"] = state
        if public_encryption_bundle is not None:
            existing["public_encryption_bundle"] = public_encryption_bundle
        peers_data[peer] = existing
        self._write_peers_json(peers_data)

    def incoming_requests(self) -> list[str]:
        """Get list of pending peer requests.

        Scans for syft_datasite_#version#{self}_*  folders NOT owned by self — those are
        peers who created folders for our datasite. Filters out already-accepted
        or rejected peers from SYFT_peers.json.
        """
        folders = self._layout.client.list_all_files(
            f"name contains '{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#' "
            f"and name contains '#{self._layout.email}#' "
            f"and trashed=false "
            f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
            f"and not 'me' in owners",
            fields="id,name,owners",
        )

        all_folder_peers = set()
        for f in folders:
            try:
                folder = GdriveP2PFolder.from_name(f["name"])
            except ValueError:
                # The query matches a name prefix, so a folder with another shape
                # can appear here.
                continue
            # Any account can make a folder with this name. The request is from
            # the owner, so the name must name the owner.
            owners = f.get("owners", [])
            if (
                len(owners) != 1
                or owners[0].get("emailAddress", "").casefold()
                != folder.peer_email.casefold()
            ):
                continue
            if folder.datasite_email == self._layout.email:
                all_folder_peers.add(folder.peer_email)

        peers_data = self.read_peer_registry()
        pending_peers = []
        for peer_email in all_folder_peers:
            undecided = peers_data.get(peer_email, {}).get("state") not in [
                "accepted",
                "rejected",
            ]
            if undecided:
                pending_peers.append(peer_email)

        return pending_peers

    def _get_own_peer_info_file_id(self) -> Optional[str]:
        """Find the own peer info file, SYFT_version.json, in /SyftBox."""
        syftbox_folder_id = self._layout.get_syftbox_folder_id()
        query = f"name='{SYFT_VERSION_FILE}' and '{syftbox_folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self._layout.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def publish_peer_info(self, peer_info: "PeerInfo") -> None:
        """Write the peer info file to /SyftBox. Creates or updates the file."""

        syftbox_folder_id = self._layout.get_syftbox_folder_id()
        file_id = self._get_own_peer_info_file_id()

        # Convert to JSON string
        json_data = peer_info.to_json()
        file_payload, _ = self._layout.client.create_file_payload(json_data)

        if file_id is None:
            # Create new file
            file_metadata = {
                "name": SYFT_VERSION_FILE,
                "parents": [syftbox_folder_id],
            }
            self._layout.client.create_syft_file(file_metadata, file_payload)
        else:
            # Update existing file
            execute_with_retries(
                self._layout.client.drive_service.files().update(
                    fileId=file_id, media_body=file_payload
                )
            )

    def _get_peer_info_file_id_of(self, peer_email: str) -> Optional[str]:
        """Find the peer info file, SYFT_version.json, of ``peer_email``."""
        query = (
            f"name='{SYFT_VERSION_FILE}' and trashed=false and '{peer_email}' in owners"
        )
        results = execute_with_retries(
            self._layout.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def read_own_peer_info(self) -> Optional["PeerInfo"]:
        """Read the own peer info file from /SyftBox.

        Returns None when there is no file.

        Raises:
            TransportUnavailableError: when Drive cannot be reached.
            ValueError: when the file is not a peer info file this client reads.
        """
        return self._read_peer_info_file(
            self._get_own_peer_info_file_id, "the own peer info file"
        )

    def read_peer_info(self, peer: str) -> Optional["PeerInfo"]:
        """Read the peer info file from a peer's /SyftBox folder.

        Returns None when the peer has no file that this account may read.

        Raises:
            TransportUnavailableError: when Drive cannot be reached.
            ValueError: when the file is not a peer info file this client reads.
        """
        return self._read_peer_info_file(
            lambda: self._get_peer_info_file_id_of(peer),
            f"the peer info file of {peer}",
        )

    def _read_peer_info_file(
        self, find_file_id: Callable[[], str | None], file_label: str
    ) -> Optional["PeerInfo"]:
        """Find, download and parse a peer info file.

        Returns None when ``find_file_id`` finds no file. ``file_label`` names
        the file in errors.
        """
        from syft.sync.version.peer_info import PeerInfo

        with raise_unavailable(f"Could not read {file_label} from Drive."):
            file_id = find_file_id()
            if file_id is None:
                return None
            file_data = self._layout.client.download_file(file_id)
        try:
            return PeerInfo.from_json(file_data.decode("utf-8"))
        except (ValueError, MigrationError) as e:
            raise ValueError(f"Could not read {file_label}: {e}") from e

    def grant_peer_info_read(self, peer: str, peer_info: "PeerInfo") -> None:
        """Share the peer info file with a peer so they can read it.

        A missing file is created from ``peer_info``, the info of this client.
        """
        file_id = self._get_own_peer_info_file_id()
        if file_id is None:
            self.publish_peer_info(peer_info)
            file_id = self._get_own_peer_info_file_id()

        if file_id:
            self._layout.client.add_permission(file_id, peer, write=False)

    def _find_encryption_bundles_folder_id(self) -> str | None:
        """The id of the encryption bundles folder of this user, or None when it does not exist."""
        if self._encryption_bundles_folder_id:
            return self._encryption_bundles_folder_id
        folder_name = GdriveEncryptionBundlesFolder(
            email=self._layout.email
        ).as_string()
        folder_id = self._layout.client.find_folder_by_name(
            folder_name, parent_id=self._layout.get_syftbox_folder_id()
        )
        if folder_id:
            self._encryption_bundles_folder_id = folder_id
        return folder_id

    def _get_or_create_encryption_bundles_folder_id(self) -> str:
        """Get or create the encryption bundles folder for this user."""
        folder_id = self._find_encryption_bundles_folder_id()
        if not folder_id:
            folder_name = GdriveEncryptionBundlesFolder(
                email=self._layout.email
            ).as_string()
            folder_id = self._layout.client.create_folder(
                folder_name, self._layout.get_syftbox_folder_id()
            )
            self._encryption_bundles_folder_id = folder_id
        return folder_id

    def _encryption_bundle_filename(self, owner_email: str, peer_email: str) -> str:
        return f"encryption_bundle_{owner_email}_for_{peer_email}.json"

    def publish_key_bundle(self, peer: str, bundle_json: str) -> None:
        """Write own encryption bundle for a peer to own bundles folder."""
        folder_id = self._get_or_create_encryption_bundles_folder_id()
        filename = self._encryption_bundle_filename(self._layout.email, peer)
        file_payload, _ = self._layout.client.create_file_payload(bundle_json)

        # Check if file already exists
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self._layout.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])

        if items:
            execute_with_retries(
                self._layout.client.drive_service.files().update(
                    fileId=items[0]["id"], media_body=file_payload
                )
            )
        else:
            file_metadata = {"name": filename, "parents": [folder_id]}
            self._layout.client.create_syft_file(file_metadata, file_payload)

    def grant_key_bundle_read(self, peer: str) -> None:
        """Share the bundles folder with a peer so they can read bundles."""
        folder_id = self._get_or_create_encryption_bundles_folder_id()
        self._layout.client.add_permission(folder_id, peer, write=False)

    def read_own_key_bundle(self, peer: str) -> str | None:
        """Read the bundle this client published for ``peer``, or None.

        A read never creates the bundles folder: without it, nothing is published.
        """
        folder_id = self._find_encryption_bundles_folder_id()
        if folder_id is None:
            return None
        filename = self._encryption_bundle_filename(self._layout.email, peer)
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = execute_with_retries(
            self._layout.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        if not items:
            return None
        return self._layout.client.download_file(items[0]["id"]).decode("utf-8")

    def read_peer_key_bundle(self, peer: str) -> str | None:
        """Read encryption bundle that peer wrote for us.

        Searches for: encryption_bundle_{peer}_for_{self._layout.email}.json
        in peer's bundles folder.
        """
        filename = self._encryption_bundle_filename(peer, self._layout.email)
        query = f"name='{filename}' and trashed=false and '{peer}' in owners"
        results = execute_with_retries(
            self._layout.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        if not items:
            return None
        try:
            data = self._layout.client.download_file(items[0]["id"])
        except Exception as e:
            print(f"Warning: could not download the bundle of {peer}: {e}")
            return None
        try:
            return data.decode("utf-8")
        except ValueError as e:
            print(f"Warning: could not read the bundle of {peer}: {e}")
            return None
