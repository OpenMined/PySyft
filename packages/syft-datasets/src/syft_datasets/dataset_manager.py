import re
import shutil
from pathlib import Path
from uuid import UUID, uuid4
from typing_extensions import Self

import yaml

from .types import PathLike, to_path
from syft_notebook_ui.types import TableList
from typing_extensions import Literal

from syft_datasets.dataset import Dataset, PrivateDatasetConfig
from syft_datasets.file_utils import copy_dir_contents, copy_paths, is_empty_dir

from .url import SyftBoxURL
from .config import SyftBoxConfig
from .permissions import set_mock_dataset_permissions, set_private_dataset_permissions
from .config import METADATA_FILENAME

DATASET_COLLECTION_PREFIX = "syft_datasetcollection"
PRIVATE_DATASET_COLLECTION_PREFIX = "syft_privatecollection"
SHARE_WITH_ANY = "any"


class SyftDatasetManager:
    def __init__(self, syftbox_folder_path: PathLike, email: str):
        self.syftbox_config = SyftBoxConfig(
            syftbox_folder=to_path(syftbox_folder_path), email=email
        )

    @classmethod
    def from_config(cls, config: SyftBoxConfig) -> Self:
        return cls(syftbox_folder_path=config.syftbox_folder, email=config.email)

    def _validate_dataset_name(self, dataset_name: str) -> None:
        # Returns True if the dataset is a valid path name on unix or windows.
        if not re.match(r"^[\w-]+$", dataset_name):
            raise ValueError(
                f"Invalid dataset name '{dataset_name}'. Only alphanumeric characters, underscores, and hyphens are allowed."
            )

    def _prepare_mock_data(self, dataset_name: str, src_path: Path) -> list[Path]:
        target_mock_dir = self.syftbox_config.get_my_mock_dataset_dir(
            dataset_name=dataset_name
        )
        # Validate src data
        if not src_path.exists():
            raise FileNotFoundError(f"Could not find mock data at {src_path}")

        if (src_path / METADATA_FILENAME).exists():
            raise ValueError(
                f"Mock data at {src_path} contains reserved file {METADATA_FILENAME}. Please rename it and try again."
            )

        # Validate dir we're making on Syftbox
        if target_mock_dir.exists() and not is_empty_dir(target_mock_dir):
            raise FileExistsError(
                f"Mock dir {target_mock_dir} already exists and is not empty."
            )
        target_mock_dir.mkdir(parents=True, exist_ok=True)

        copied_files = []
        if src_path.is_dir():
            copied_files = copy_dir_contents(
                src=src_path,
                dst=target_mock_dir,
                exists_ok=True,
            )
        elif src_path.is_file():
            copied_files = copy_paths(
                files=[src_path],
                dst=target_mock_dir,
                exists_ok=True,
            )
        else:
            raise ValueError(
                f"Mock data path {src_path} must be an existing file or directory."
            )

        return copied_files

    def _prepare_private_data(
        self,
        dataset_name: str,
        src_path: Path,
    ) -> list[Path]:
        private_dir = self.syftbox_config.private_dir_for_my_dataset(
            dataset_name=dataset_name
        )
        private_dir.mkdir(parents=True, exist_ok=True)

        copied_files = []
        if src_path.is_dir():
            # TODO: Implementing without copying private data to `SyftBox/private``
            copied_files = copy_dir_contents(
                src=src_path,
                dst=private_dir,
                exists_ok=True,
            )
        elif src_path.is_file():
            copied_files = copy_paths(
                files=[src_path],
                dst=private_dir,
                exists_ok=True,
            )
        else:
            raise ValueError(
                f"Private data path {src_path} must be an existing file or directory."
            )

        return copied_files

    def _prepare_private_config(
        self,
        dataset_name: str,
        dataset_uid: UUID,
    ) -> None:
        """
        The private dataset config is used to store private metadata separately from the public dataset metadata.
        """
        private_metadata_path: Path = (
            self.syftbox_config.private_metadata_filename_for_my_dataset(
                dataset_name=dataset_name
            )
        )
        if private_metadata_path.exists():
            raise FileExistsError(
                f"Private metadata file {private_metadata_path} already exists."
            )

        private_config = PrivateDatasetConfig(
            uid=dataset_uid,
            data_dir=private_metadata_path.parent,
        )

        private_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        private_config.save(filepath=private_metadata_path)

    def _prepare_readme(self, dataset_name: str, src_file: Path | None) -> list[Path]:
        target_mock_dir = self.syftbox_config.get_my_mock_dataset_dir(
            dataset_name=dataset_name
        )
        copied_files = []
        if src_file is not None:
            if not src_file.is_file():
                raise FileNotFoundError(f"Could not find README at {src_file}")
            if not src_file.suffix.lower() == ".md":
                raise ValueError("readme file must be a markdown (.md) file.")
            copied_files = copy_paths(
                files=[src_file],
                dst=target_mock_dir,
                exists_ok=True,
            )
        return copied_files

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
        # copy_private_data: bool = True, # TODO
    ) -> Dataset:
        """_summary_

        Args:
            name (str): Unique of the dataset to create.
            mock_path (PathLike): Path to the existing mock data. This can be a file or a directory.
            private_path (PathLike): Path to the existing private data. This can be a file or a directory.
            summary (str | None, optional): Short summary of the dataset. Defaults to None.
            readme_path (Path | None, optional): Markdown README in the public dataset. Defaults to None.
            location (str | None, optional): Location identifier for the dataset, e.g. 'high-side-1234'.
                Only required for datasets that are hosted on a remote location and require manual syncing.
                Defaults to None.
            tags (list[str] | None, optional): Optional tags for the dataset. Defaults to None.
            users (list[str] | str | None, optional): Users to share dataset with. Can be list of emails, SHARE_WITH_ANY, or None (default, share with no one).

        Returns:
            Dataset: The created Dataset object.
        """
        mock_path: Path = to_path(mock_path)
        private_path: Path = to_path(private_path)
        readme_path: Path | None = to_path(readme_path) if readme_path else None
        tags = tags or []

        mock_url = self.syftbox_config.get_mock_url_for_my_dataset(dataset_name=name)
        readme_url = (
            self.syftbox_config.get_readme_url_for_my_dataset(
                dataset_name=name, readme_name=readme_path.name
            )
            if readme_path
            else None
        )

        # Generate private_url for the dataset Private URLs use a simple path format
        private_url = self.syftbox_config.get_private_url_for_my_dataset(
            dataset_name=name
        )

        # Prepare mock data and collect file paths
        all_mock_file_paths = self._prepare_mock_data(
            dataset_name=name,
            src_path=mock_path,
        )

        # Mock files exclude dataset.yaml and readme.md
        # Convert absolute paths to SyftBoxURLs

        # Prepare readme and collect file paths
        readme_files = self._prepare_readme(
            dataset_name=name,
            src_file=readme_path,
        )

        public_metadata_path = (
            self.syftbox_config.public_metadata_filename_for_my_dataset(
                dataset_name=name
            )
        )

        mock_file_paths = [
            f
            for f in all_mock_file_paths
            if f != public_metadata_path and f not in readme_files
        ]

        mock_files_urls = [
            SyftBoxURL.from_path(
                path=file_path,
                syftbox_folder=self.syftbox_config.syftbox_folder,
            )
            for file_path in mock_file_paths
        ]

        self._prepare_private_data(
            dataset_name=name,
            src_path=private_path,
        )

        # TODO enable adding private data without copying to SyftBox
        # e.g. private_data_dir = dataset._private_metadata_dir if copy_private_data else private_path
        dataset_uid = uuid4()
        self._prepare_private_config(dataset_uid=dataset_uid, dataset_name=name)

        dataset = Dataset(
            uid=dataset_uid,
            name=name,
            mock_url=mock_url,
            private_url=private_url,
            readme_url=readme_url,
            summary=summary,
            location=location,
            tags=tags,
            mock_files_urls=mock_files_urls,
            _syftbox_config=self.syftbox_config,
        )
        # needs to set since its a private attr
        dataset._syftbox_config = self.syftbox_config

        # Save dataset metadata
        dataset.save(filepath=public_metadata_path)
        # Set permissions on mock and private directories
        self._set_new_dataset_permissions(dataset=dataset, users=users)
        return dataset

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

    def _load_dataset_from_dir(self, dataset_dir: Path) -> Dataset:
        metadata_path = dataset_dir / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}")

        return Dataset.load(
            filepath=metadata_path,
            syftbox_config=self.syftbox_config,
        )

    def get(self, name: str, datasite: str | None = None) -> Dataset:
        datasite = datasite or self.syftbox_config.email
        mock_dir = self.syftbox_config.get_mock_dataset_dir(
            dataset_name=name,
            datasite=datasite,
        )

        if not mock_dir.exists():
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
        return self._load_dataset_from_dir(mock_dir)

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

    def get_all(
        self,
        datasite: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
        sort_order: Literal["asc", "desc"] = "asc",
    ) -> list[Dataset]:
        all_datasets = []

        if datasite:
            datasites_to_check = [datasite]
        else:
            syftbox_folder = self.syftbox_config.syftbox_folder
            # All directories with "@" in the name are peer/owner email directories
            datasites_to_check = [
                d.name for d in syftbox_folder.iterdir() if d.is_dir() and "@" in d.name
            ]

        for datasite in datasites_to_check:
            public_datasets_dir = self.syftbox_config.public_datasets_dir_for_datasite(
                datasite
            )
            if not public_datasets_dir.exists():
                continue
            for dataset_dir in public_datasets_dir.iterdir():
                if dataset_dir.is_dir():
                    try:
                        dataset = self._load_dataset_from_dir(dataset_dir)
                        all_datasets.append(dataset)
                    except Exception:
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

        # Delete the dataset directories
        if dataset.mock_dir.exists():
            shutil.rmtree(dataset.mock_dir)
        if dataset._private_metadata_dir.exists():
            shutil.rmtree(dataset._private_metadata_dir)

    def get_private_dataset_files(self, name: str) -> dict[Path, bytes]:
        """Get private dataset files as {path_in_datasite: content}.

        Returns paths relative to the datasite (e.g. private/syft_datasets/{name}/{file}).
        For private_metadata.yaml, clears data_dir before including it.
        """
        dataset = self.get(name=name, datasite=self.syftbox_config.email)
        private_dir = dataset.private_dir
        if not private_dir.exists():
            raise ValueError(f"Private data directory not found: {private_dir}")

        files = {}
        for f in private_dir.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(private_dir)
            path_in_datasite = Path(f"private/syft_datasets/{name}") / rel
            if f.name == "private_metadata.yaml" and rel == Path(f.name):
                files[path_in_datasite] = self._private_config_without_data_dir(f)
            else:
                files[path_in_datasite] = f.read_bytes()

        if not files:
            raise ValueError(f"No private files found for dataset '{name}'")
        return files

    def _private_config_without_data_dir(self, config_path: Path) -> bytes:
        """Load private_metadata.yaml and return it with data_dir cleared."""
        config = PrivateDatasetConfig.load(filepath=config_path)
        config.data_dir = Path("")
        return yaml.safe_dump(
            config.model_dump(mode="json"), indent=2, sort_keys=False
        ).encode()
