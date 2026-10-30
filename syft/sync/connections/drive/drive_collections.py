"""Collections on Google Drive: folders in /SyftBox shared by Drive permission.

A collection folder is named ``{prefix}{variant}_{tag}_{content_hash}``.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from syft.sync.connections.base_connection import FileCollection
from syft.sync.connections.drive.gdrive_client import GOOGLE_FOLDER_MIME_TYPE
from syft.sync.connections.drive.drive_cached_client import DriveCachedClient
from syft.sync.connections.drive.gdrive_retry import execute_with_retries
from syft.sync.connections.protocols import CollectionsConnection


def collection_name_query(prefix: str) -> str:
    """A Drive query that finds a collection of ``prefix`` in any layout.

    It has no trailing '_', because a layout writes its variant in that
    position. A client that predates multi-layout searches with the '_', and so
    never lists a layout it cannot read.
    """
    return f"name contains '{prefix}'"


@lru_cache(maxsize=None)
def _collection_name_re(prefix: str) -> "re.Pattern[str]":
    """Matches '{prefix}{variant}_{tag}_{hash}'.

    The variant and the hash hold no '_'. The tag can hold one, so it takes
    every character in between.
    """
    return re.compile(
        rf"^{re.escape(prefix)}"
        r"(?P<variant>[^_]*)_(?P<tag>.+)_(?P<content_hash>[^_]+)$"
    )


class CollectionFolder(BaseModel):
    """Naming value object for collection folders (domain-agnostic).

    Collections (datasets, private datasets, or any future collection type) are
    named ``{prefix}{variant}_{tag}_{content_hash}`` on the wire. The variant
    names the layout (see CollectionSyncSpec) and is "" for the original one, so
    that name is unchanged byte for byte. This is the single source of truth for
    building and parsing the name; it also computes the content hash from the
    folder's file contents.

    A writer normally folds the variant into ``prefix`` (the spec builds the
    qualified prefix once) and leaves ``variant`` empty; ``from_name`` splits
    them apart again for a reader.
    """

    prefix: str
    tag: str
    content_hash: str
    variant: str = ""

    @property
    def folder_name(self) -> str:
        """The on-wire folder name. Doubles as the cache key (same string)."""
        return f"{self.prefix}{self.variant}_{self.tag}_{self.content_hash}"

    @classmethod
    def from_name(cls, prefix: str, name: str) -> "CollectionFolder":
        """Parse '{prefix}{variant}_{tag}_{hash}' into a CollectionFolder.

        Raises ValueError if the name does not carry the prefix, so callers can
        skip non-matching folders.
        """
        match = _collection_name_re(prefix).match(name)
        if match is None:
            raise ValueError(f"Invalid collection folder name: {name}")
        return cls(prefix=prefix, **match.groupdict())

    @staticmethod
    def compute_hash(files: dict[str, bytes]) -> str:
        """Compute a content hash from file contents."""
        from syft.sync.file_utils import compute_file_hashes

        return compute_file_hashes(files)


class DriveCollectionsConnection(CollectionsConnection):
    """Collection folders in /SyftBox, shared by Drive permission."""

    def __init__(self, cached_client: DriveCachedClient):
        self._cached_client = cached_client
        # folder name -> id of my collection folder
        self.collection_folder_id_cache: dict[str, str] = {}

    def clone_for_thread(self) -> "DriveCollectionsConnection":
        """Collections over a new Drive client, for use from another thread,
        with these caches."""
        clone = DriveCollectionsConnection(self._cached_client.clone_for_thread())
        clone.collection_folder_id_cache = dict(self.collection_folder_id_cache)
        return clone

    def reset_caches(self) -> None:
        self._cached_client.reset_caches()
        self.collection_folder_id_cache.clear()

    def create(self, prefix: str, tag: str, content_hash: str, owner_email: str) -> str:
        """Create /SyftBox/{prefix}_{tag}_{hash} folder."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name

        # Check cache
        if folder_name in self.collection_folder_id_cache:
            return self.collection_folder_id_cache[folder_name]

        syftbox_folder_id = self._cached_client.get_syftbox_folder_id()

        # Check if exists
        folder_id = self._cached_client.client.find_folder_by_name(
            folder_name, parent_id=syftbox_folder_id
        )
        if folder_id:
            self.collection_folder_id_cache[folder_name] = folder_id
            return folder_id

        # Create new folder
        folder_id = self._cached_client.client.create_folder(
            folder_name, syftbox_folder_id
        )
        self.collection_folder_id_cache[folder_name] = folder_id
        return folder_id

    def mark_shared_with_any(self, prefix: str, tag: str, content_hash: str) -> None:
        """Mark collection as shared with 'any' via appProperties."""
        folder_id = self._get_collection_folder_id(prefix, tag, content_hash)
        execute_with_retries(
            self._cached_client.client.drive_service.files().update(
                fileId=folder_id,
                body={"appProperties": {"syft_shared_with_any": "true"}},
            )
        )

    def grant_read(
        self, prefix: str, tag: str, content_hash: str, users: list[str]
    ) -> None:
        """Share collection folder with specific users via batch API."""
        if not users:
            return
        folder_id = self._get_collection_folder_id(prefix, tag, content_hash)
        self._cached_client.client.batch_add_permissions(folder_id, users)

    def upload(
        self, prefix: str, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        """Upload files to a collection folder."""
        folder_id = self._get_collection_folder_id(prefix, tag, content_hash)

        for file_path, content in files.items():
            file_payload, _ = self._cached_client.client.create_file_payload(content)
            file_name = Path(file_path).name

            file_metadata = {"name": file_name, "parents": [folder_id]}
            self._cached_client.client.create_syft_file(file_metadata, file_payload)

    def list_mine(self, prefix: str) -> list[str]:
        """List collections created by DO (owned by me). Returns list of tags.

        A tag appears once however many layouts hold it.
        """
        syftbox_folder_id = self._cached_client.get_syftbox_folder_id()
        query = (
            f"{collection_name_query(prefix)} and '{syftbox_folder_id}' in parents "
            f"and 'me' in owners and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        result: list[str] = []
        for folder in self._cached_client.client.list_all_files(query, fields="name"):
            try:
                cf = CollectionFolder.from_name(prefix, folder["name"])
            except ValueError:
                continue
            if cf.tag not in result:
                result.append(cf.tag)
        return result

    def list_mine_with_permissions(
        self,
        prefix: str,
    ) -> list[FileCollection]:
        """List all DO's collections with permissions info, one per layout."""
        syftbox_folder_id = self._cached_client.get_syftbox_folder_id()
        query = (
            f"{collection_name_query(prefix)} and '{syftbox_folder_id}' in parents "
            f"and 'me' in owners and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        collections = []
        for folder in self._cached_client.client.list_all_files(
            query, fields="id,name,appProperties"
        ):
            try:
                cf = CollectionFolder.from_name(prefix, folder["name"])
            except ValueError:
                continue
            has_anyone = (
                folder.get("appProperties", {}).get("syft_shared_with_any") == "true"
            )
            collections.append(
                FileCollection(
                    folder_id=folder["id"],
                    tag=cf.tag,
                    content_hash=cf.content_hash,
                    has_any_permission=has_anyone,
                    variant=cf.variant,
                )
            )

        return collections

    def delete(self, prefix: str, tag: str) -> None:
        """Delete every layout of ``tag`` published under ``prefix``."""
        collections = self.list_mine_with_permissions(prefix)
        for c in collections:
            if c.tag == tag:
                self._cached_client.client.delete_file_by_id(c.folder_id)
                folder_name = CollectionFolder(
                    prefix=prefix,
                    tag=c.tag,
                    content_hash=c.content_hash,
                    variant=c.variant,
                ).folder_name
                self.collection_folder_id_cache.pop(folder_name, None)

    def delete_by_handle(self, handle: str) -> None:
        self._cached_client.client.delete_file_by_id(handle)
        self.collection_folder_id_cache = {
            name: folder_id
            for name, folder_id in self.collection_folder_id_cache.items()
            if folder_id != handle
        }

    def list_shared_with_me(self, prefix: str) -> list[dict]:
        """List collections shared with DS (not owned by me), one per layout.

        Returns list of dicts with keys: owner_email, tag, content_hash, variant
        """
        query = (
            f"{collection_name_query(prefix)} and not 'me' in owners "
            f"and trashed=false and mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        )
        result = []
        for folder in self._cached_client.client.list_all_files(
            query, fields="name, owners"
        ):
            try:
                cf = CollectionFolder.from_name(prefix, folder["name"])
            except ValueError:
                # Skip folders that don't match the expected format
                continue
            owner_email = folder.get("owners", [{}])[0].get("emailAddress", "unknown")
            result.append(
                {
                    "owner_email": owner_email,
                    "tag": cf.tag,
                    "content_hash": cf.content_hash,
                    "variant": cf.variant,
                }
            )
        return result

    def download(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> dict[str, bytes]:
        """Download all files from a collection."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name
        # Try to find folder by name (could be owned by someone else)
        folder_id = self._cached_client.client.find_folder_by_name(
            folder_name, owner_email=owner_email
        )

        if not folder_id:
            raise ValueError(f"Collection {tag} with hash {content_hash} not found")

        file_metadatas = self._cached_client.get_file_metadatas_from_folder(folder_id)
        files = {}
        for file_meta in file_metadatas:
            file_id = file_meta["id"]
            file_name = file_meta["name"]
            files[file_name] = self._cached_client.client.download_file(file_id)

        return files

    def list_files(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> List[Dict]:
        """Get file metadata from a collection without downloading."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name
        folder_id = self._cached_client.client.find_folder_by_name(
            folder_name, owner_email=owner_email
        )

        if not folder_id:
            raise ValueError(f"Collection {tag} with hash {content_hash} not found")

        file_metadatas = self._cached_client.get_file_metadatas_from_folder(folder_id)
        return [{"file_id": f["id"], "file_name": f["name"]} for f in file_metadatas]

    def fetch_file(self, handle: str) -> bytes:
        """Download a single file from a collection."""
        return self._cached_client.client.download_file(handle)

    def _get_collection_folder_id(
        self, prefix: str, tag: str, content_hash: str
    ) -> str:
        """Get folder ID for a collection, with caching."""
        folder_name = CollectionFolder(
            prefix=prefix, tag=tag, content_hash=content_hash
        ).folder_name
        if folder_name in self.collection_folder_id_cache:
            return self.collection_folder_id_cache[folder_name]

        syftbox_folder_id = self._cached_client.get_syftbox_folder_id()
        folder_id = self._cached_client.client.find_folder_by_name(
            folder_name, parent_id=syftbox_folder_id
        )

        if not folder_id:
            raise ValueError(
                f"Collection folder {tag} with hash {content_hash} not found"
            )

        self.collection_folder_id_cache[folder_name] = folder_id
        return folder_id
