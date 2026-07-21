import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
from uuid import UUID, uuid4

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    MigrationService,
    ProtocolSchema,
)
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from .config import (
    METADATA_FILENAME,
    PRIVATE_METADATA_FILENAME,
    SyftBoxConfig,
    is_protocol_dir_name,
)
from .dataset_ref import DatasetNotFoundError, DatasetRef, PrivateConfigNotFoundError
from .file_utils import copy_dir_contents, copy_paths, is_empty_dir
from .migrations.registry import DATASET_PROTOCOL_VERSION, dataset_registry
from .models import Dataset, PrivateDatasetConfig
from .protocolcodecs import CODECS, ProtocolCodec
from .url import SyftBoxURL

__all__ = [
    "DatasetRef",
    "DatasetNotFoundError",
    "PrivateConfigNotFoundError",
    "DatasetStorage",
    "DatasetSourceFiles",
]

# On-disk files that are part of the dataset representation, not payload data;
# skipped when re-copying an existing dataset into a new layout during migration.
_NON_PAYLOAD_FILES = frozenset(
    {METADATA_FILENAME, PRIVATE_METADATA_FILENAME, PERMISSION_FILE_NAME}
)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class DatasetSourceFiles:
    """Source paths a caller hands to ``create_dataset`` (ephemeral input).

    Unversioned by design: these paths never persist. The on-disk layout is
    versioned by the codecs' DatasetConfig, and the persisted file manifest rides
    the versioned Dataset (``mock_files_urls``). A new file category is just a new
    field here plus a new Dataset version + codec layout entry.
    """

    mock: Path  # file or directory
    private: Path  # file or directory
    readme: Optional[Path] = None  # .md file


@dataclass(frozen=True)
class _DatasetFields:
    """Layout-free dataset fields, built once and shared across every version written."""

    uid: UUID
    created_at: datetime
    updated_at: datetime
    name: str
    summary: Optional[str] = None
    location: Optional[str] = None
    tags: list[str] = field(default_factory=list)


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

    # -- create / migrate ----------------------------------------------------
    def create_dataset(
        self,
        *,
        name: str,
        source: DatasetSourceFiles,
        uid: Optional[UUID] = None,
        summary: Optional[str] = None,
        location: Optional[str] = None,
        tags: Optional[list[str]] = None,
        peer_emails: Optional[list[str]] = None,
        protocol_versions: Optional[list[str]] = None,
    ) -> dict[str, Dataset]:
        """Write a new dataset in every protocol version its audience can read.

        Copies the source files into each version's on-disk layout and writes the
        metadata/private config. Returns {protocol_version: written Dataset}.

        By default the versions are inferred from ``peer_emails`` (no/unknown peers
        => the widest-compatible protocol). Pass ``protocol_versions`` to write
        exactly those versions instead, skipping inference.
        """
        self.validate_dataset_name(name)
        if source.mock.is_dir() and (source.mock / METADATA_FILENAME).exists():
            raise ValueError(
                f"Mock data at {source.mock} contains reserved file "
                f"{METADATA_FILENAME}. Please rename it and try again."
            )
        if protocol_versions is None:
            protocol_versions = list(
                self.target_protocol_versions_for_peers(peer_emails)
            )
        now = _utcnow()
        fields = _DatasetFields(
            uid=uid or uuid4(),
            created_at=now,
            updated_at=now,
            name=name,
            summary=summary,
            location=location,
            tags=tags or [],
        )
        created: dict[str, Dataset] = {}
        for protocol_version in sorted(protocol_versions, key=int):
            ref = self.new_dataset_ref(name, protocol_version)
            created[protocol_version] = self._materialize_version(ref, fields, source)
        return created

    def migrate_dataset(self, ref: DatasetRef, target_protocol_version: str) -> Dataset:
        """Rewrite an existing on-disk dataset into another protocol layout.

        Copies the source layout's files into the target layout and writes the
        metadata/private config there, preserving identity (uid/timestamps).
        Does not delete the source copy. Owner-only.
        """
        if ref.owner != self.config.email:
            raise ValueError("Can only migrate datasets you own.")
        old = self.read_dataset(ref)
        source = DatasetSourceFiles(
            mock=self.public_dataset_dir(ref),
            private=self.private_dataset_dir(ref),
            readme=old.readme_path,
        )
        fields = _DatasetFields(
            uid=old.uid,
            created_at=old.created_at,
            updated_at=old.updated_at,
            name=old.name,
            summary=old.summary,
            location=old.location,
            tags=old.tags,
        )
        target_ref = self.new_dataset_ref(ref.name, target_protocol_version)
        return self._materialize_version(
            target_ref, fields, source, exclude_names=_NON_PAYLOAD_FILES
        )

    def _materialize_version(
        self,
        ref: DatasetRef,
        fields: _DatasetFields,
        source: DatasetSourceFiles,
        exclude_names: frozenset[str] = frozenset(),
    ) -> Dataset:
        """Copy files + write metadata/config for a single protocol version."""
        target_mock_dir = self.public_dataset_dir(ref)
        private_dir = self.private_dataset_dir(ref)

        all_mock_file_paths = self._copy_mock_data(ref, source.mock, exclude_names)
        readme_files = self._copy_readme(ref, source.readme)

        # mock_files_urls is the copied mock payload minus the metadata file and
        # the readme (which peers read via their own URL fields).
        public_metadata_path = target_mock_dir / METADATA_FILENAME
        mock_file_paths = [
            f
            for f in all_mock_file_paths
            if f != public_metadata_path and f not in readme_files
        ]

        dataset = Dataset(
            uid=fields.uid,
            created_at=fields.created_at,
            updated_at=fields.updated_at,
            name=fields.name,
            mock_url=self._url_for(target_mock_dir),
            private_url=self._url_for(private_dir),
            readme_url=(
                self._url_for(target_mock_dir / source.readme.name)
                if source.readme
                else None
            ),
            summary=fields.summary,
            location=fields.location,
            tags=fields.tags,
            mock_files_urls=[self._url_for(f) for f in mock_file_paths],
        )
        dataset._syftbox_config = self.config
        dataset._ref = ref

        self._copy_private_data(ref, source.private, exclude_names)

        self.write_dataset_metadata(ref, dataset)
        self.write_private_config(
            ref, PrivateDatasetConfig(uid=fields.uid, data_dir=private_dir)
        )
        return dataset

    def _url_for(self, path: Path) -> SyftBoxURL:
        return SyftBoxURL.from_path(
            path=path, syftbox_folder=self.config.syftbox_folder
        )

    def _copy_mock_data(
        self,
        ref: DatasetRef,
        src_path: Path,
        exclude_names: frozenset[str] = frozenset(),
    ) -> list[Path]:
        target_dir = self.public_dataset_dir(ref)
        if not src_path.exists():
            raise FileNotFoundError(f"Could not find mock data at {src_path}")
        if target_dir.exists() and not is_empty_dir(target_dir):
            raise FileExistsError(
                f"Mock dir {target_dir} already exists and is not empty."
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            return copy_dir_contents(
                src=src_path,
                dst=target_dir,
                exists_ok=True,
                exclude_names=exclude_names,
            )
        if src_path.is_file():
            return copy_paths(files=[src_path], dst=target_dir, exists_ok=True)
        raise ValueError(
            f"Mock data path {src_path} must be an existing file or directory."
        )

    def _copy_private_data(
        self,
        ref: DatasetRef,
        src_path: Path,
        exclude_names: frozenset[str] = frozenset(),
    ) -> list[Path]:
        private_target_dir = self.private_dataset_dir(ref)
        private_target_dir.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            # TODO: Implementing without copying private data to `SyftBox/private``
            return copy_dir_contents(
                src=src_path,
                dst=private_target_dir,
                exists_ok=True,
                exclude_names=exclude_names,
            )
        if src_path.is_file():
            return copy_paths(files=[src_path], dst=private_target_dir, exists_ok=True)
        raise ValueError(
            f"Private data path {src_path} must be an existing file or directory."
        )

    def _copy_readme(self, ref: DatasetRef, src_file: Optional[Path]) -> list[Path]:
        if src_file is None:
            return []
        if not src_file.is_file():
            raise FileNotFoundError(f"Could not find README at {src_file}")
        if not src_file.suffix.lower() == ".md":
            raise ValueError("readme file must be a markdown (.md) file.")
        return copy_paths(
            files=[src_file], dst=self.public_dataset_dir(ref), exists_ok=True
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

    # -- scanning ------------------------------------------------------------
    def iter_dataset_refs_all_protocols(
        self, datasite_email: str
    ) -> Iterator[DatasetRef]:
        """Yield a ref per on-disk dataset copy, across every protocol layout.

        A dataset broadcast to a mixed audience is written once per protocol, so
        the same (owner, name) can appear multiple times. Storage-internal; use
        it when you must touch every copy (e.g. deletion). Callers outside
        storage should use ``iter_dataset_refs``.
        """
        for codec in self.codecs:
            yield from codec.iter_dataset_refs(datasite_email)

    def iter_dataset_refs(self, datasite_email: str) -> Iterator[DatasetRef]:
        """Yield one ref per dataset, in its preferred (newest) protocol layout.

        Collapses the per-protocol copies of a dataset to the single layout a
        reader should use, so callers see each dataset once.
        """
        best: dict[tuple[str, str], DatasetRef] = {}
        for ref in self.iter_dataset_refs_all_protocols(datasite_email):
            key = (ref.owner, ref.name)
            current = best.get(key)
            if current is None or int(ref.protocol_version) > int(
                current.protocol_version
            ):
                best[key] = ref
        yield from best.values()

    def find_dataset_ref(self, datasite_email: str, name: str) -> DatasetRef:
        """The ref for ``name`` in a datasite, in its preferred protocol layout."""
        for ref in self.iter_dataset_refs(datasite_email):
            if ref.name == name:
                return ref
        raise DatasetNotFoundError(f"Dataset '{name}' not found")

    # -- deletion ------------------------------------------------------------
    def delete_dataset(self, datasite_email: str, name: str) -> list[Path]:
        """Remove every on-disk copy of a dataset, across all protocol layouts.

        A dataset may be written in several protocol versions at once, so this
        removes both the public dataset dir and the private metadata dir for each
        version present on disk. Returns the directories that were removed.
        """
        refs = [
            ref
            for ref in self.iter_dataset_refs_all_protocols(datasite_email)
            if ref.name == name
        ]
        removed: list[Path] = []
        for ref in refs:
            for directory in (
                self.public_dataset_dir(ref),
                self.private_dataset_dir(ref),
            ):
                if directory.exists():
                    shutil.rmtree(directory)
                    removed.append(directory)
        return removed

    # -- model IO ------------------------------------------------------------
    def read_dataset(self, ref: DatasetRef) -> Dataset:
        """Load a dataset's dataset.yaml, upgraded to the latest version."""
        codec = self._codec_for(ref.protocol_version)
        data = codec.read(codec.metadata_path(ref), "Dataset")
        dataset = self._upgrade(data, "Dataset")
        # Name comes from the path (the dataset dir), never a spoofable file.
        dataset.name = ref.name
        dataset._syftbox_config = self.config
        dataset._ref = ref
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

    def write_dataset_metadata(self, ref: DatasetRef, dataset: Dataset) -> Path:
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
