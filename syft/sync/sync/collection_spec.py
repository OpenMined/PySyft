"""The collection-sync primitive.

``CollectionSyncSpec`` is the domain-free contract that tells the generic sync
engine HOW to sync a "collection" (a named, content-hashed folder like a dataset's
mock or private data) — WITHOUT the engine knowing what a dataset is. The concrete
specs are supplied by the domain layer (e.g. ``syft_rds.config.DATASET_COLLECTION_SPECS``).
"""

from pathlib import Path

from pydantic import BaseModel, model_validator


class CollectionLayout(BaseModel):
    """One on-the-wire layout of a collection.

    An owner can publish the same collection several times, once per layout, so
    that peers of different ages each find one they can read. ``variant`` is the
    discriminator the owner writes into the folder name directly after the
    prefix; the original layout uses "" and so keeps the name it always had.
    ``local_subpath`` is where that layout lands under the owner's datasite.

    The engine treats ``variant`` as opaque. What it means (a dataset protocol
    version, say) is the domain's business.
    """

    variant: str = ""
    local_subpath: Path


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
    # Every layout this client can read, oldest first. When an owner publishes a
    # collection in several layouts, the watcher keeps the last one in this list
    # that the owner published, and skips the rest. Defaults to the single
    # original layout.
    layouts: list[CollectionLayout] = []

    @model_validator(mode="after")
    def _default_layout(self) -> "CollectionSyncSpec":
        if not self.layouts:
            self.layouts = [CollectionLayout(local_subpath=self.local_subpath)]
        return self

    def wire_prefix(self, variant: str) -> str:
        """The folder-name prefix an owner writes for one layout.

        The variant sits between the prefix and the '_' that starts the tag, so
        a client that predates multi-layout searches for '<prefix>_' and never
        lists a layout it cannot read.
        """
        return f"{self.prefix}{variant}"

    def layout_for(self, variant: str) -> CollectionLayout | None:
        """The layout for a wire variant, or None when this client cannot read it."""
        return next((la for la in self.layouts if la.variant == variant), None)

    def rank_of(self, variant: str) -> int:
        """Position of a variant in ``layouts``; -1 when unreadable. Higher is newer."""
        return next(
            (i for i, la in enumerate(self.layouts) if la.variant == variant), -1
        )

    @classmethod
    def public(
        cls,
        prefix: str,
        local_subpath: "Path",
        layouts: list[CollectionLayout] | None = None,
    ) -> "CollectionSyncSpec":
        """A shareable, mirrored collection (e.g. a dataset's mock data).

        Peers' watchers pull it, and it re-downloads whenever the owner
        republishes changed content.
        """
        return cls(
            prefix=prefix,
            local_subpath=local_subpath,
            immutable=False,
            owner_only=False,
            layouts=layouts or [],
        )

    @classmethod
    def private(
        cls,
        prefix: str,
        local_subpath: "Path",
        layouts: list[CollectionLayout] | None = None,
    ) -> "CollectionSyncSpec":
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
            layouts=layouts or [],
        )
