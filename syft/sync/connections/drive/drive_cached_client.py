"""Where syft keeps its folders in one account's Google Drive.

``DriveCachedClient`` finds and creates syft's folders and caches their ids. A
folder whose name holds the client version follows the version rules in this
module.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, field_validator

from syft.sync.connections.drive.gdrive_client import (
    GOOGLE_FOLDER_MIME_TYPE,
    GDriveClient,
)
from syft.sync.connections.drive.gdrive_retry import execute_with_retries
from syft.sync.events.file_change_event import FileChangeEventsMessageFileName
from syft.sync.transport.naming import timestamp_from_message_name
from syft.sync.version.version_info import _parse_semver
from syft.version import SYFT_VERSION

SYFTBOX_FOLDER = "SyftBox"
GDRIVE_P2P_FOLDER_DATASITE_PREFIX = "syft_datasite"


class GdriveArchiveFolder(BaseModel):
    sender_email: str
    recipient_email: str

    def as_string(self) -> str:
        return f"syft_{self.sender_email}_to_{self.recipient_email}_archive"


class GdrivePersonalSyftboxFolder(BaseModel):
    """Folder for personal SyftBox: {version}#{email} under /SyftBox/"""

    email: str

    def as_string(self) -> str:
        return f"{SYFT_VERSION}#{self.email}"


# An email part of a folder name goes into Drive queries and local paths. It has
# one '@', a '.' in the domain, and no quote, slash, '#', whitespace or control
# character. It is not RFC 5322.
_EMAIL_PART = r"[^@\s'\"\\/#\x00-\x1f\x7f]+"
_FOLDER_EMAIL_RE = re.compile(rf"{_EMAIL_PART}@{_EMAIL_PART}\.{_EMAIL_PART}")


class GdriveP2PFolder(BaseModel):
    """Folder for peer communication: syft_datasite#version#datasite_email#inbox|outbox#peer_email"""

    datasite_email: str
    folder_type: str  # "inbox" or "outbox"
    peer_email: str

    @field_validator("datasite_email", "peer_email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        if not _FOLDER_EMAIL_RE.fullmatch(v):
            raise ValueError(f"Invalid email in P2P folder name: {v!r}")
        return v

    def as_string(self) -> str:
        return f"{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#{SYFT_VERSION}#{self.datasite_email}#{self.folder_type}#{self.peer_email}"

    @classmethod
    def from_name(cls, name: str) -> "GdriveP2PFolder":
        parts = name.split("#")
        if len(parts) != 5 or parts[0] != GDRIVE_P2P_FOLDER_DATASITE_PREFIX:
            raise ValueError(f"Invalid P2P folder name: {name}")
        return cls(datasite_email=parts[2], folder_type=parts[3], peer_email=parts[4])


# Helpers for finding folders whose names embed SYFT_VERSION. Folder
# names use '#' or '-' as field separators with the version as one field,
# so we walk those fields looking for an X.Y.Z-shaped chunk -- no per-format
# parser needed. Patch-compat means same major.minor (matches the protocol
# definition in version_info.is_compatible_with).


def _looks_like_version(s: str) -> bool:
    """True if s is shaped like 'X.Y.Z' with all numeric parts."""
    parts = s.split(".")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def _extract_version_from_name(name: str) -> str | None:
    """Return the first '#'/'-'-separated field that looks like a semver."""
    for chunk in name.replace("-", "#").split("#"):
        if _looks_like_version(chunk):
            return chunk
    return None


# A folder id and name, with the version from the name. The version fields come
# first, so the default sort puts these in version order.
_VersionedFolder = tuple[int, int, int, str, str]


def _sorted_by_version(folders: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Folders from the lowest version to the highest.

    A name with no readable version sorts first, so a versioned folder always
    wins when the caller takes the last entry.
    """

    def key(entry: tuple[str, str]) -> tuple[int, int, int]:
        version_str = _extract_version_from_name(entry[1])
        if version_str is None:
            return (-1, -1, -1)
        try:
            return _parse_semver(version_str)
        except ValueError:
            return (-1, -1, -1)

    return sorted(folders, key=key)


def _partition_by_version(
    folders: list[tuple[str, str]],
    current_version: str | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Split folders into (compatible, older, newer) by the version in the name.

    Compatible means the same major and minor as the current version. The function
    drops a folder that has no version in its name. Each list starts at the lowest
    version.
    """
    if current_version is None:
        current_version = SYFT_VERSION
    try:
        current = _parse_semver(current_version)
    except ValueError:
        return [], [], []

    compatible: list[_VersionedFolder] = []
    older: list[_VersionedFolder] = []
    newer: list[_VersionedFolder] = []
    for fid, name in folders:
        version_str = _extract_version_from_name(name)
        if version_str is None:
            continue
        try:
            found = _parse_semver(version_str)
        except ValueError:
            continue
        entry = (*found, fid, name)
        if found[:2] == current[:2]:
            compatible.append(entry)
        elif found < current:
            older.append(entry)
        else:
            newer.append(entry)

    def _ordered(entries: list[_VersionedFolder]) -> list[tuple[str, str]]:
        return [(fid, name) for *_, fid, name in sorted(entries)]

    return _ordered(compatible), _ordered(older), _ordered(newer)


def valid_events_from_file_metadatas(
    file_metadatas: list[dict],
) -> list[FileChangeEventsMessageFileName]:
    res = []
    for file_metadata in file_metadatas:
        fname = file_metadata["name"]
        try:
            message_filename = FileChangeEventsMessageFileName.from_string(fname)
        except ValueError:
            print(f"Warning: invalid event file name: {fname}")
            continue
        res.append(message_filename)
    return res


@dataclass
class DriveCachedClient:
    """The Drive client of one account, and where syft's folders are in it.

    It finds and creates syft's folders, follows the version rules for a
    versioned folder name, and caches the folder ids so a lookup costs no
    Drive call the second time.
    """

    client: GDriveClient
    email: str
    # /SyftBox: the top folder, with the inboxes, outboxes and personal folder.
    _syftbox_folder_id: str | None = field(default=None, init=False)
    # /SyftBox/{version}#{email}: the personal data.
    _personal_syftbox_folder_id: str | None = field(default=None, init=False)
    # peer email -> id of the folder I made for that peer's datasite.
    peer_datasite_inbox_cache: dict[str, str] = field(default_factory=dict, init=False)
    peer_datasite_outbox_cache: dict[str, str] = field(default_factory=dict, init=False)
    # peer email -> id of the folder that peer made for my datasite.
    own_datasite_inbox_cache: dict[str, str] = field(default_factory=dict, init=False)
    own_datasite_outbox_cache: dict[str, str] = field(default_factory=dict, init=False)
    # sender email -> id of my archive folder for that sender.
    archive_folder_id_cache: dict[str, str] = field(default_factory=dict, init=False)

    def clone_for_thread(self) -> "DriveCachedClient":
        """A copy over a new Drive client, for use from another thread, that
        knows the folder ids this one found."""
        clone = DriveCachedClient(self.client.clone_for_thread(), self.email)
        self.copy_caches_to(clone)
        return clone

    def copy_caches_to(self, other: "DriveCachedClient") -> None:
        """Seed the folder ids of ``other`` from this one."""
        other._syftbox_folder_id = self._syftbox_folder_id
        other._personal_syftbox_folder_id = self._personal_syftbox_folder_id
        other.peer_datasite_inbox_cache = dict(self.peer_datasite_inbox_cache)
        other.peer_datasite_outbox_cache = dict(self.peer_datasite_outbox_cache)
        other.own_datasite_inbox_cache = dict(self.own_datasite_inbox_cache)
        other.own_datasite_outbox_cache = dict(self.own_datasite_outbox_cache)
        other.archive_folder_id_cache = dict(self.archive_folder_id_cache)

    def reset_caches(self) -> None:
        self._syftbox_folder_id = None
        self._personal_syftbox_folder_id = None
        self.peer_datasite_inbox_cache.clear()
        self.peer_datasite_outbox_cache.clear()
        self.own_datasite_inbox_cache.clear()
        self.own_datasite_outbox_cache.clear()
        self.archive_folder_id_cache.clear()

    def get_syftbox_folder_id(self) -> str:
        """/SyftBox"""
        # cached
        if self._syftbox_folder_id:
            return self._syftbox_folder_id
        else:
            syftbox_folder_id = self.get_syftbox_folder_id_from_drive()
            if syftbox_folder_id:
                self._syftbox_folder_id = syftbox_folder_id
                return self._syftbox_folder_id
            else:
                return self.create_syftbox_folder()

    def get_syftbox_folder_id_from_drive(self) -> str | None:
        query = f"name='{SYFTBOX_FOLDER}' and mimeType='application/vnd.google-apps.folder' and 'me' in owners and trashed=false"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id, name)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def create_syftbox_folder(self) -> str:
        """Creates /SyftBox"""
        folder = self.client.create_syft_file(
            {"name": SYFTBOX_FOLDER, "mimeType": GOOGLE_FOLDER_MIME_TYPE}
        )
        return folder["id"]

    def get_personal_syftbox_folder_id(self) -> str:
        """/SyftBox/{version}#{email}"""
        if self._personal_syftbox_folder_id:
            return self._personal_syftbox_folder_id
        folders = self.client.find_folders(
            name_contains=[f"#{self.email}"],
            parent_id=self.get_syftbox_folder_id(),
            owner_email=self.email,
        )
        # The substring '#{email}' also matches p2p folder names that end in
        # '#{peer}#{type}#{email}'. Personal folder shape is exactly
        # '{version}#{email}', so require a single '#'.
        folders = [(fid, name) for fid, name in folders if name.count("#") == 1]
        folder_id = self.find_or_adopt_versioned_folder(
            folders,
            current_name=GdrivePersonalSyftboxFolder(email=self.email).as_string(),
        )
        if folder_id:
            self._personal_syftbox_folder_id = folder_id
            return folder_id
        return self.create_personal_syftbox_folder()

    def create_personal_syftbox_folder(self) -> str:
        """Creates /SyftBox/{version}#{email}"""
        syftbox_folder_id = self.get_syftbox_folder_id()
        folder_name = GdrivePersonalSyftboxFolder(email=self.email).as_string()
        return self.client.create_folder(folder_name, syftbox_folder_id)

    def find_or_adopt_versioned_folder(
        self,
        folders: list[tuple[str, str]],
        current_name: str,
        current_version: str | None = None,
    ) -> str | None:
        """Return the id of a PRIVATE folder for this client version, or None.

        A private folder name holds the client version, so a minor upgrade looks
        for a name that does not exist yet. This method renames the folder of the
        highest earlier version to `current_name` and keeps the data. A new folder
        would leave the data of the user on Drive and out of reach.

        Renames the folder, so the caller must own it and no peer may look it up by
        name. A P2P folder fails both conditions: use `expect_one` for those.

        Raises RuntimeError if only a folder from a later version exists, or if
        more than one compatible folder exists.
        """
        compatible, older, newer = _partition_by_version(folders, current_version)
        if compatible:
            return self.expect_one(compatible)
        if newer:
            names = [n for _, n in newer]
            latest = _extract_version_from_name(names[-1])
            raise RuntimeError(
                f"Found a folder from a later client version on Drive: {names}. "
                f"This client is {current_version or SYFT_VERSION} and "
                f"cannot read that data. Install syft {latest} or later."
            )
        if not older:
            return None

        folder_id, name = older[-1]
        execute_with_retries(
            self.client.drive_service.files().update(
                fileId=folder_id, body={"name": current_name}
            )
        )
        print(f"Adopted the folder of an earlier version: {name} -> {current_name}")
        if len(older) > 1:
            stale = [n for _, n in older[:-1]]
            print(
                f"Warning: {len(stale)} folder(s) of earlier versions stay on "
                f"Drive: {stale}"
            )
        return folder_id

    def expect_one(self, folders: list[tuple[str, str]]) -> str | None:
        """Return the single folder id, None if empty, raise if more than one.

        Auto-picking from multiple matches risks orphaning data, so the user
        must reconcile on Drive themselves.
        """
        if not folders:
            return None
        if len(folders) == 1:
            return folders[0][0]
        names = [n for _, n in folders]
        raise RuntimeError(
            f"Found {len(folders)} compatible folders on Drive: {names}. "
            f"Exactly one is expected. This usually indicates a stale folder "
            f"left behind by a prior client version. Please delete the stale "
            f"folder(s) on Drive (keeping the one with your data) and retry."
        )

    def create_peer_datasite_folder(self, peer_email: str, folder_type: str) -> str:
        """Create a datasite folder under /SyftBox for peer's datasite."""
        if folder_type not in ("inbox", "outbox"):
            raise ValueError(
                f"Invalid folder_type: {folder_type}. Must be 'inbox' or 'outbox'."
            )
        parent_id = self.get_syftbox_folder_id()
        folder = GdriveP2PFolder(
            datasite_email=peer_email, folder_type=folder_type, peer_email=self.email
        )
        folder_name = folder.as_string()
        folder_id = self.client.create_folder(folder_name, parent_id)
        if folder_type == "inbox":
            self.peer_datasite_inbox_cache[peer_email] = folder_id
        else:
            self.peer_datasite_outbox_cache[peer_email] = folder_id
        return folder_id

    def find_p2p_folder_id(
        self,
        datasite_email: str,
        folder_type: str,
        peer_email: str,
        owner_email: str,
    ) -> str | None:
        folders = self.client.find_folders(
            name_contains=[
                f"{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}#",
                f"#{datasite_email}#{folder_type}#{peer_email}",
            ],
            owner_email=owner_email,
        )

        # Drive's `name contains` is a fuzzy token/prefix match, not a substring
        # match, so it can return folders for other peers when email tokens
        # collide
        def _is_exact_match(name: str) -> bool:
            try:
                folder = GdriveP2PFolder.from_name(name)
            except ValueError:
                return False
            return (
                folder.datasite_email == datasite_email
                and folder.folder_type == folder_type
                and folder.peer_email == peer_email
            )

        # Ignore the version in the name. Each peer builds this name from its own
        # client version, so a filter here hides the folder that the peer uses.
        # After an upgrade the client therefore finds the old folder and writes to
        # it. It makes no second folder, which an older peer would never look for.
        candidates = _sorted_by_version(
            [(fid, name) for fid, name in folders if _is_exact_match(name)]
        )
        if not candidates:
            return None
        if len(candidates) > 1:
            print(
                f"Warning: {len(candidates)} P2P folders for {datasite_email} "
                f"{folder_type} {peer_email}; using {candidates[-1][1]}"
            )
        return candidates[-1][0]

    def get_peer_datasite_inbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{peer}_inbox_{self}, owned by self."""
        if peer_email in self.peer_datasite_inbox_cache:
            return self.peer_datasite_inbox_cache[peer_email]
        folder_id = self.find_p2p_folder_id(
            datasite_email=peer_email,
            folder_type="inbox",
            peer_email=self.email,
            owner_email=self.email,
        )
        if folder_id is not None:
            self.peer_datasite_inbox_cache[peer_email] = folder_id
        return folder_id

    def get_peer_datasite_outbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{peer}_outbox_{self}, owned by self."""
        if peer_email in self.peer_datasite_outbox_cache:
            return self.peer_datasite_outbox_cache[peer_email]
        folder_id = self.find_p2p_folder_id(
            datasite_email=peer_email,
            folder_type="outbox",
            peer_email=self.email,
            owner_email=self.email,
        )
        if folder_id is not None:
            self.peer_datasite_outbox_cache[peer_email] = folder_id
        return folder_id

    def get_own_datasite_inbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{self}_inbox_{peer}, owned by peer."""
        if peer_email in self.own_datasite_inbox_cache:
            return self.own_datasite_inbox_cache[peer_email]
        folder_id = self.find_p2p_folder_id(
            datasite_email=self.email,
            folder_type="inbox",
            peer_email=peer_email,
            owner_email=peer_email,
        )
        if folder_id is not None:
            self.own_datasite_inbox_cache[peer_email] = folder_id
        return folder_id

    def get_own_datasite_outbox_id(self, peer_email: str) -> str | None:
        """Get folder: syft_datasite_{self}_outbox_{peer}, owned by peer."""
        if peer_email in self.own_datasite_outbox_cache:
            return self.own_datasite_outbox_cache[peer_email]
        folder_id = self.find_p2p_folder_id(
            datasite_email=self.email,
            folder_type="outbox",
            peer_email=peer_email,
            owner_email=peer_email,
        )
        if folder_id is not None:
            self.own_datasite_outbox_cache[peer_email] = folder_id
        return folder_id

    def get_archive_folder_id(self, sender_email: str) -> str:
        if sender_email not in self.archive_folder_id_cache:
            archive_folder_id = self.get_archive_folder_id_from_drive(
                sender_email
            ) or self.create_archive_folder(sender_email)
            self.archive_folder_id_cache[sender_email] = archive_folder_id
        return self.archive_folder_id_cache[sender_email]

    def get_archive_folder_id_from_drive(self, sender_email: str) -> str | None:
        archive_folder = GdriveArchiveFolder(
            sender_email=sender_email, recipient_email=self.email
        )
        archive_folder_name = archive_folder.as_string()
        query = f"name='{archive_folder_name}' and mimeType='application/vnd.google-apps.folder' and 'me' in owners and trashed=false"
        results = execute_with_retries(
            self.client.drive_service.files().list(q=query, fields="files(id)")
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def create_archive_folder(self, sender_email: str) -> str:
        archive_folder = GdriveArchiveFolder(
            sender_email=sender_email, recipient_email=self.email
        )
        archive_folder_name = archive_folder.as_string()
        syftbox_folder_id = self.get_syftbox_folder_id()
        return self.client.create_folder(archive_folder_name, syftbox_folder_id)

    def get_file_metadatas_from_folder(
        self,
        folder_id: str,
        since_timestamp: float | None = None,
        page_size: int = 100,
    ) -> list[dict]:
        """
        Get file metadatas from folder with early termination.

        Args:
            folder_id: Google Drive folder ID
            since_timestamp: Optional timestamp. If provided, only files with a
                           newer timestamp are returned, and pagination stops at
                           a file from an earlier second.
            page_size: Number of files to fetch per API call. Default 100.

        Returns:
            List of file metadata dicts, sorted by name descending (newest first)
        """
        query = f"'{folder_id}' in parents and trashed=false"
        all_files = []
        page_token = None

        while True:
            results = execute_with_retries(
                self.client.drive_service.files().list(
                    q=query,
                    fields="files(id, name, size, mimeType, modifiedTime), nextPageToken",
                    pageSize=page_size,
                    pageToken=page_token,
                    orderBy="name desc",
                )
            )

            page_files = results.get("files", [])

            # Early termination: Check if this page contains old files
            if since_timestamp is not None and page_files:
                should_stop = False

                for file in page_files:
                    timestamp = timestamp_from_message_name(file["name"])

                    if timestamp is not None:
                        if timestamp > since_timestamp:
                            all_files.append(file)
                        elif int(timestamp) < int(since_timestamp):
                            # Drive sorts names as strings, so a 0.9 fraction of
                            # a second lists before 0.95. We stop when the
                            # second flips.
                            should_stop = True
                            break
                    else:
                        # No timestamp in filename, include the file
                        all_files.append(file)

                if should_stop:
                    # Don't fetch more pages
                    break
            else:
                # No early termination check, add all files
                all_files.extend(page_files)

            # Check for next page
            page_token = results.get("nextPageToken")
            if not page_token:
                break

        return all_files
