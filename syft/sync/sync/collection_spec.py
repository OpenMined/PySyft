"""The collection-sync primitive.

``CollectionSyncSpec`` is the domain-free contract that tells the generic sync
engine HOW to sync a "collection" (a named, content-hashed folder like a dataset's
mock or private data) — WITHOUT the engine knowing what a dataset is. The concrete
specs are supplied by the domain layer (e.g. ``syft_rds.config.DATASET_COLLECTION_SPECS``).
"""

from pathlib import Path

from pydantic import BaseModel


class CollectionSyncSpec(BaseModel):
    prefix: str
    # Subpath from owner_email to the collection folder,
    # e.g. Path("public/syft_datasets")
    local_subpath: Path
    # Download authority. False = mirror: re-download when the remote content-hash
    # changes (the remote is the source of truth, e.g. published mock data).
    # True = restore-only: download only when absent locally and never overwrite the
    # local copy (the local copy is the source of truth, e.g. the owner's real data).
    immutable: bool = False
    # Sharing policy. False = shareable: a peer's watcher may pull it (e.g. mock data).
    # True = owner-only: never shared with peers; the owner restores it for itself and
    # peer-facing watchers skip it entirely (e.g. the owner's private data backup).
    owner_only: bool = False

    @classmethod
    def public(cls, prefix: str, local_subpath: "Path") -> "CollectionSyncSpec":
        """A shareable, mirrored collection (e.g. a dataset's mock data).

        Peers' watchers pull it, and it re-downloads whenever the owner
        republishes changed content.
        """
        return cls(
            prefix=prefix,
            local_subpath=local_subpath,
            immutable=False,
            owner_only=False,
        )

    @classmethod
    def private(cls, prefix: str, local_subpath: "Path") -> "CollectionSyncSpec":
        """An owner-only, restore-only collection (e.g. a dataset's real data).

        Never shared with peers, peer-facing watchers skip it, and it is only
        restored to the owner when absent locally (the local copy is authoritative
        and is never overwritten by the backup).
        """
        return cls(
            prefix=prefix,
            local_subpath=local_subpath,
            immutable=True,
            owner_only=True,
        )
