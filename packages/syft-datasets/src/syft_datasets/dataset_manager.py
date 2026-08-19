from pathlib import Path

from typing_extensions import Self

import yaml
from syft_migration import ProtocolSchema

from .types import PathLike, to_path
from syft_notebook_ui.types import TableList
from typing_extensions import Literal

from syft_datasets.models import Dataset
from syft_datasets.dataset_ref import DatasetRef
from syft_datasets.dataset_storage import DatasetSourceFiles, DatasetStorage

from .config import PRIVATE_METADATA_FILENAME, SyftBoxConfig
from .permissions import set_mock_dataset_permissions, set_private_dataset_permissions

DATASET_COLLECTION_PREFIX = "syft_datasetcollection"
PRIVATE_DATASET_COLLECTION_PREFIX = "syft_privatecollection"
SHARE_WITH_ANY = "any"


class SyftDatasetManager:
    def __init__(
        self,
        syftbox_folder_path: PathLike,
        email: str,
        peer_schemas: dict[str, ProtocolSchema] | None = None,
    ):
        self.syftbox_config = SyftBoxConfig(
            syftbox_folder=to_path(syftbox_folder_path), email=email
        )
        # peer_schemas (peer email -> dataset ProtocolSchema): syft-client
        # passes PeerManager's live map here (updated in place as peer version
        # files load). Peers without an entry resolve to the widest-compatible
        # protocol, so datasets stay readable by unknown-version peers.
        self.storage = DatasetStorage(
            config=self.syftbox_config, peer_schemas=peer_schemas
        )

    @classmethod
    def from_config(
        cls,
        config: SyftBoxConfig,
        peer_schemas: dict[str, ProtocolSchema] | None = None,
    ) -> Self:
        return cls(
            syftbox_folder_path=config.syftbox_folder,
            email=config.email,
            peer_schemas=peer_schemas,
        )

    def create(
        self,
        name: str,
        mock_path: PathLike,
        private_path: PathLike,
        summary: str | None = None,
        readme_path: Path | None = None,
        location: str | None = None,
        tags: list[str] | None = None,
        users: list[str] | str | None = None,
        protocol_versions: list[str] | None = None,
        # copy_private_data: bool = True, # TODO
    ) -> Dataset:
        """Create a dataset, writing it in each protocol version its audience can read.

        Args:
            name (str): Unique name of the dataset to create.
            mock_path (PathLike): Path to the existing mock data (file or directory).
            private_path (PathLike): Path to the existing private data (file or directory).
            summary (str | None, optional): Short summary of the dataset. Defaults to None.
            readme_path (Path | None, optional): Markdown README in the public dataset.
            location (str | None, optional): Location identifier for the dataset.
            tags (list[str] | None, optional): Optional tags for the dataset.
            users (list[str] | str | None, optional): Users to share dataset with. Can be
                a list of emails, SHARE_WITH_ANY, or None (default, share with no one).
            protocol_versions (list[str] | None, optional): Write exactly these
                protocol versions instead of inferring them from ``users``.

        Returns:
            Dataset: The created Dataset object (the newest protocol version written).
        """
        created = self.create_all(
            name=name,
            mock_path=mock_path,
            private_path=private_path,
            summary=summary,
            readme_path=readme_path,
            location=location,
            tags=tags,
            users=users,
            protocol_versions=protocol_versions,
        )
        # Return the newest protocol version written (richest layout).
        return created[max(created, key=int)]

    def create_all(
        self,
        name: str,
        mock_path: PathLike,
        private_path: PathLike,
        summary: str | None = None,
        readme_path: Path | None = None,
        location: str | None = None,
        tags: list[str] | None = None,
        users: list[str] | str | None = None,
        protocol_versions: list[str] | None = None,
    ) -> dict[str, "Dataset"]:
        """Create a dataset and return every protocol copy it wrote.

        Same as ``create``, but returns {protocol_version: Dataset} instead of
        one copy. A caller that puts the dataset on a transport needs them all,
        because each copy goes to the peers that read its layout.
        """
        source = DatasetSourceFiles(
            mock=to_path(mock_path),
            private=to_path(private_path),
            readme=to_path(readme_path) if readme_path else None,
        )
        created = self.storage.create_dataset(
            name=name,
            source=source,
            summary=summary,
            location=location,
            tags=tags,
            peer_emails=self._peer_emails(users),
            protocol_versions=protocol_versions,
        )
        for dataset in created.values():
            self._set_new_dataset_permissions(dataset=dataset, users=users)
        return created

    def migrate(
        self,
        name: str,
        to_version: str,
        users: list[str] | str | None = None,
    ) -> Dataset:
        """Rewrite an owned dataset into another protocol layout, re-applying permissions.

        Storage copies the files + writes metadata for the new layout; the manager
        re-applies read permissions. The audience (``users``) must be supplied by
        the caller as it is on create — granted readers are not recoverable from
        disk via the permissions API.
        """
        ref = self.storage.find_dataset_ref(self.syftbox_config.email, name)
        migrated = self.storage.migrate_dataset(ref, to_version)
        self._set_new_dataset_permissions(dataset=migrated, users=users)
        return migrated

    @staticmethod
    def _peer_emails(users: list[str] | str | None) -> list[str] | None:
        """Audience emails for protocol negotiation; None means no/any peers."""
        if users is None or users == SHARE_WITH_ANY:
            return None
        if isinstance(users, str):
            return [users]
        return list(users)

    def _set_new_dataset_permissions(
        self, dataset: Dataset, users: list[str] | str | None
    ) -> None:
        if users is None:
            users = []
        mock_user_permissions = (
            ["*"] if users == SHARE_WITH_ANY else (users if users else [])
        )
        if mock_user_permissions:
            set_mock_dataset_permissions(
                self.syftbox_config.syftbox_folder,
                self.syftbox_config.email,
                dataset.mock_dir,
                mock_user_permissions,
            )
        set_private_dataset_permissions(
            self.syftbox_config.syftbox_folder,
            self.syftbox_config.email,
            dataset.private_dir,
        )

    def get(self, name: str, datasite: str | None = None) -> Dataset:
        datasite = datasite or self.syftbox_config.email
        try:
            ref = self.storage.find_dataset_ref(datasite, name)
        except FileNotFoundError:
            available = self.get_all()
            if available:
                listing = "\n".join(
                    f"   • {d.name} (from {d.owner})" for d in available
                )
            else:
                listing = "   (none found — check your peer connections)"
            raise FileNotFoundError(
                f"❌ Dataset '{name}' not found in {datasite}'s datasite.\n\n"
                f"   Possible reasons:\n"
                f"   1. The DO hasn't created this dataset yet.\n"
                f"   2. You're not connected to them as a peer.\n"
                f"   3. You need to sync first — try: client.sync()\n\n"
                f"   Available datasets:\n"
                f"{listing}"
            )
        return self.storage.read_dataset(ref)

    def __getitem__(self, key: str | int) -> Dataset:
        if isinstance(key, int):
            return self.get_all()[key]
        return self.get(name=key)

    def __len__(self) -> int:
        return len(self.get_all())

    def __iter__(self):
        return iter(self.get_all())

    def __repr__(self) -> str:
        datasets = self.get_all()
        return f"SyftDatasetManager({len(datasets)} datasets)"

    def _repr_html_(self) -> str:
        from .dataset_manager_repr import dataset_manager_repr_html

        return dataset_manager_repr_html(self.get_all())

    def _all_syftbox_datasites(self) -> list[str]:
        syftbox_folder = self.syftbox_config.syftbox_folder
        # All directories with "@" in the name are peer/owner email directories
        return [
            d.name for d in syftbox_folder.iterdir() if d.is_dir() and "@" in d.name
        ]

    def get_all(
        self,
        datasite: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
        sort_order: Literal["asc", "desc"] = "asc",
    ) -> list[Dataset]:
        # storage.iter_dataset_refs already yields one ref per dataset in its
        # preferred protocol layout, so there is nothing to dedupe here.
        datasites = (
            [datasite] if datasite is not None else self._all_syftbox_datasites()
        )

        all_datasets = []
        for datasite in datasites:
            for ref in self.storage.iter_dataset_refs(datasite):
                try:
                    all_datasets.append(self.storage.read_dataset(ref))
                except Exception:
                    print(
                        f"Error reading dataset {ref.name} from {ref.owner}, skipping",
                    )
                    continue

        if order_by is not None:
            all_datasets.sort(
                key=lambda d: getattr(d, order_by),
                reverse=(sort_order.lower() == "desc"),
            )

        if offset is not None:
            all_datasets = all_datasets[offset:]
        if limit is not None:
            all_datasets = all_datasets[:limit]

        return TableList(all_datasets)

    def delete(
        self,
        name: str,
        datasite: str | None = None,
        require_confirmation: bool = True,
    ) -> None:
        datasite = datasite or self.syftbox_config.email

        if datasite != self.syftbox_config.email:
            # NOTE this check is easily bypassed, but bypassing does not have any effect.
            # When bypassed, the dataset will be restored because the user only has
            # read access to someone else's datasite.
            raise ValueError(
                "Cannot delete datasets from a datasite that is not your own."
            )

        try:
            dataset = self.get(
                name=name,
                datasite=datasite,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset {name} not found in datasite {datasite}")

        if require_confirmation:
            msg = (
                "Deleting this dataset will remove the following folders:\n"
                f"Mock data: {dataset.mock_dir}\n"
                f"Private metadata: {dataset._private_metadata_dir}\n"
            )
            if (
                dataset._private_metadata_dir.exists()
                and dataset.private_dir.resolve().absolute()
                == dataset._private_metadata_dir.resolve().absolute()
            ):
                msg += (
                    "WARNING: this will also delete the private data from your system\n"
                )
            else:
                msg += "Private data will not be deleted from your system, it is not managed by SyftBox.\n"

            msg += "Are you sure you want to delete these folders? (yes/no): "
            confirmation = input(msg).strip().lower()
            if confirmation != "yes":
                print("Dataset deletion cancelled.")
                return

        # Remove every on-disk copy (all protocol versions) via the storage layer.
        self.storage.delete_dataset(datasite, name)

    def get_private_dataset_files(
        self, name: str, protocol_version: str | None = None
    ) -> dict[Path, bytes]:
        """Get private dataset files as {path_in_datasite: content}.

        Returns paths relative to the datasite (e.g.
        private/syft_datasets/[v<n>/]{name}/{file}); the paths carry the copy's
        protocol layout, so ``protocol_version`` selects the copy a specific
        reader scans (the preferred/newest copy by default). For
        private_metadata.yaml, clears data_dir before including it.
        """
        datasite = self.syftbox_config.email
        ref = self.storage.find_dataset_ref(
            datasite, name, protocol_version=protocol_version
        )
        private_dir = self.storage.private_dataset_dir(ref)
        if not private_dir.exists():
            raise ValueError(f"Private data directory not found: {private_dir}")

        datasite_root = self.syftbox_config.syftbox_folder / datasite
        private_rel_root = private_dir.relative_to(datasite_root)

        files = {}
        for f in private_dir.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(private_dir)
            path_in_datasite = private_rel_root / rel
            if f.name == PRIVATE_METADATA_FILENAME and rel == Path(f.name):
                files[path_in_datasite] = self._private_config_without_data_dir(ref)
            else:
                files[path_in_datasite] = f.read_bytes()

        if not files:
            raise ValueError(f"No private files found for dataset '{name}'")
        return files

    def _private_config_without_data_dir(self, ref: DatasetRef) -> bytes:
        """Serialize the dataset's private_metadata.yaml with data_dir cleared.

        Written in the dataset's on-disk protocol format so peers can read it.
        """
        config = self.storage.read_private_config(ref)
        config.data_dir = Path("")
        data = config.disk_dict()
        return yaml.safe_dump(data, indent=2, sort_keys=False).encode()
