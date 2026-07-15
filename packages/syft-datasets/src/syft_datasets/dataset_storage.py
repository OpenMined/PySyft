from pathlib import Path
from typing import Iterator, Optional

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    MigrationService,
    ProtocolSchema,
)

from .config import SyftBoxConfig, is_protocol_dir_name
from .dataset_ref import DatasetNotFoundError, DatasetRef, PrivateConfigNotFoundError
from .migrations.registry import DATASET_PROTOCOL_VERSION, dataset_registry
from .models import Dataset, PrivateDatasetConfig
from .protocolcodecs import CODECS, ProtocolCodec

__all__ = [
    "DatasetRef",
    "DatasetNotFoundError",
    "PrivateConfigNotFoundError",
    "DatasetStorage",
]


class DatasetStorage:
    """Dataset filesystem IO and path resolution, delegating disk layout to codecs.

    DatasetStorage owns migration: reads upgrade objects to the latest registered
    version in memory; writes downgrade them to what a reading peer understands.
    Each on-disk storage format lives behind a ProtocolCodec, selected by the
    protocol version (from a dataset's on-disk layout for reads; the negotiated
    version for new datasets). The last codec is the current protocol.

    Unlike jobs (each copy targets one peer), a public dataset is a single copy
    read by the whole audience, so new datasets are written at the version(s) the
    audience can read (see ``target_protocol_versions_for_peers``), defaulting to
    the widest-compatible (oldest) protocol when no peers are known.
    """

    def __init__(
        self,
        config: SyftBoxConfig,
        registry: MigrationRegistry = dataset_registry,
        peer_schemas: Optional[dict[str, ProtocolSchema]] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.service = MigrationService(registry=registry)
        # peer email -> dataset ProtocolSchema; filled in by syft-client later.
        # Peers without an entry cannot be assumed to read the current layout, so
        # they resolve to the widest-compatible (oldest) protocol.
        self.peer_schemas: dict[str, ProtocolSchema] = peer_schemas or {}
        self.codecs = [cls(config) for cls in CODECS]

    @property
    def _codec_by_protocol_version(self) -> dict[str, ProtocolCodec]:
        # protocol version -> codec; one codec may serve several versions.
        return {
            protocol_version: codec
            for codec in self.codecs
            for protocol_version in codec.protocol_versions
        }

    def _codec_for(self, protocol_version: str) -> ProtocolCodec:
        codec = self._codec_by_protocol_version.get(protocol_version)
        if codec is None:
            raise MigrationError(
                f"No codec for dataset protocol version {protocol_version!r}"
            )
        return codec

    @property
    def _widest_protocol_version(self) -> str:
        """The oldest protocol version any current client can read (widest compat)."""
        return min(self._codec_by_protocol_version, key=int)

    # -- peers / protocol ----------------------------------------------------
    def negotiated_protocol_version_for_peer(
        self, peer_email: str, raise_on_unknown: bool = True
    ) -> str:
        """The dataset protocol version to speak with ``peer_email``.

        Negotiated as the minimum of our own protocol version and the peer's, so
        both sides use a version they can read. A peer without a known schema
        raises by default; with ``raise_on_unknown=False`` it is assumed to run
        the current protocol.
        """
        schema = self.peer_schemas.get(peer_email)
        if schema is not None:
            return min(DATASET_PROTOCOL_VERSION, schema.version, key=int)
        if raise_on_unknown:
            raise MigrationError(
                f"No dataset protocol schema known for peer {peer_email!r}"
            )
        return DATASET_PROTOCOL_VERSION

    def target_protocol_versions_for_peers(
        self, peer_emails: Optional[list[str]] = None
    ) -> set[str]:
        """The set of protocol versions to write so every peer can read a copy.

        A dataset is written once per distinct version in the audience. A known
        peer contributes ``min(ours, theirs)``; an unknown peer (or no audience)
        contributes the widest-compatible protocol, since we cannot assume it can
        read a newer layout.
        """
        if not peer_emails:
            return {self._widest_protocol_version}
        versions: set[str] = set()
        for email in peer_emails:
            schema = self.peer_schemas.get(email)
            if schema is not None:
                versions.add(min(DATASET_PROTOCOL_VERSION, schema.version, key=int))
            else:
                versions.add(self._widest_protocol_version)
        return versions

    def new_dataset_ref(self, name: str, protocol_version: str) -> DatasetRef:
        """A ref for a new dataset owned by the current user."""
        return DatasetRef(
            owner=self.config.email, name=name, protocol_version=protocol_version
        )

    # -- naming --------------------------------------------------------------
    @staticmethod
    def validate_dataset_name(dataset_name: str) -> None:
        if is_protocol_dir_name(dataset_name):
            raise ValueError(
                f"Dataset name {dataset_name!r} is reserved for protocol version directories"
            )

    # -- path resolution -----------------------------------------------------
    def public_dataset_dir(self, ref: DatasetRef) -> Path:
        return self._codec_for(ref.protocol_version).public_dataset_dir(ref)

    def private_dataset_dir(self, ref: DatasetRef) -> Path:
        return self._codec_for(ref.protocol_version).private_dataset_dir(ref)

    def metadata_path(self, ref: DatasetRef) -> Path:
        return self._codec_for(ref.protocol_version).metadata_path(ref)

    def private_metadata_path(self, ref: DatasetRef) -> Path:
        return self._codec_for(ref.protocol_version).private_metadata_path(ref)

    # -- scanning (union over all protocol layouts) --------------------------
    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]:
        """Yield a ref per dataset under public/syft_datasets/, across all codecs."""
        for codec in self.codecs:
            yield from codec.iter_dataset_refs(datasite_email)

    def find_dataset_ref(self, datasite_email: str, name: str) -> DatasetRef:
        """The ref for ``name`` in a datasite; prefers the highest protocol layout."""
        matches = [
            ref for ref in self.iter_dataset_refs(datasite_email) if ref.name == name
        ]
        if not matches:
            raise DatasetNotFoundError(f"Dataset '{name}' not found")
        return max(matches, key=lambda r: int(r.protocol_version))

    # -- model IO ------------------------------------------------------------
    def read_dataset(self, ref: DatasetRef) -> Dataset:
        """Load a dataset's dataset.yaml, upgraded to the latest version."""
        codec = self._codec_for(ref.protocol_version)
        data = codec.read(codec.metadata_path(ref), "Dataset")
        dataset = self._upgrade(data, "Dataset")
        # Name comes from the path (the dataset dir), never a spoofable file.
        dataset.name = ref.name
        dataset._syftbox_config = self.config
        dataset._protocol_version = ref.protocol_version
        return dataset

    def read_private_config(self, ref: DatasetRef) -> PrivateDatasetConfig:
        """Load a dataset's private_metadata.yaml, upgraded to the latest version.

        Raises PrivateConfigNotFoundError if the file does not exist.
        """
        codec = self._codec_for(ref.protocol_version)
        path = codec.private_metadata_path(ref)
        if not path.exists():
            raise PrivateConfigNotFoundError(
                f"Private config not found for dataset {ref.name}"
            )
        return self._upgrade(
            codec.read(path, "PrivateDatasetConfig"), "PrivateDatasetConfig"
        )

    def write_dataset(self, ref: DatasetRef, dataset: Dataset) -> Path:
        """Write dataset.yaml in the version/format for this ref's protocol."""
        codec = self._codec_for(ref.protocol_version)
        return self._write(codec, codec.metadata_path(ref), dataset, ref)

    def write_private_config(
        self, ref: DatasetRef, config: PrivateDatasetConfig
    ) -> Path:
        """Write private_metadata.yaml in the version/format for this ref's protocol."""
        codec = self._codec_for(ref.protocol_version)
        return self._write(codec, codec.private_metadata_path(ref), config, ref)

    # -- internals -----------------------------------------------------------
    def _upgrade(self, data: dict, canonical_name: str) -> MigratableObject:
        obj = self.service.load(data)
        return self.service.migrate(obj, self.registry.latest_version(canonical_name))

    def _write(
        self,
        codec: ProtocolCodec,
        path: Path,
        obj: MigratableObject,
        ref: DatasetRef,
    ) -> Path:
        """Downgrade ``obj`` to the ref's protocol schema, then let the codec persist it."""
        schema = self.registry.schema_for_protocol_version(ref.protocol_version)
        downgraded = self.service.migrate_to_schema(obj, schema)
        codec.write(path, downgraded)
        return path
