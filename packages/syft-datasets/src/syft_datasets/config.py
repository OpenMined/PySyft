import re
from typing import Optional

from pydantic import BaseModel, Field
from pathlib import Path
from .url import SyftBoxURL
from .migrations.registry import DATASET_PROTOCOL_VERSION

SYFT_DATASETS_FOLDER_NAME = "syft_datasets"
METADATA_FILENAME = "dataset.yaml"
PRIVATE_METADATA_FILENAME = "private_metadata.yaml"

# Datasets under public/syft_datasets/ and private/syft_datasets/ live inside a
# protocol-version segment ("v1", "v2", ...); protocol 0 (<= 0.1.20) had none.
PROTOCOL_DIR_RE = re.compile(r"^v\d+$")


def is_protocol_dir_name(name: str) -> bool:
    return PROTOCOL_DIR_RE.match(name) is not None


def protocol_dir_name(protocol_version: str) -> Optional[str]:
    """The path segment for a protocol version; None for protocol 0."""
    return None if protocol_version == "0" else f"v{protocol_version}"


class SyftBoxConfig(BaseModel):
    syftbox_folder: Path = Field(
        ..., description="Path to the SyftBox folder on the local filesystem."
    )
    email: str = Field(..., description="Email associated with the SyftBox.")

    @property
    def private_dir(self) -> Path:
        return self.syftbox_folder / self.email / "private"

    @property
    def public_dir(self) -> Path:
        return self.syftbox_folder / self.email / "public"

    def public_datasets_root_for_datasite(self, datasite: str) -> Path:
        """The syft_datasets scan root for a datasite (no protocol segment)."""
        return self.syftbox_folder / datasite / "public" / SYFT_DATASETS_FOLDER_NAME

    def private_datasets_root(self) -> Path:
        """The private syft_datasets scan root (no protocol segment)."""
        return self.private_dir / SYFT_DATASETS_FOLDER_NAME

    def private_datasets_root_for_owner(self, owner: str) -> Path:
        """The private syft_datasets scan root under an owner's datasite."""
        return self.syftbox_folder / owner / "private" / SYFT_DATASETS_FOLDER_NAME

    def get_private_dataset_dir(
        self,
        owner: str,
        dataset_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> Path:
        return self._with_segment(
            self.private_datasets_root_for_owner(owner), protocol_version, dataset_name
        )

    def public_datasets_dir_for_datasite(self, datasite: str) -> Path:
        # Backwards-compatible accessor; ensures the scan root exists.
        dir = self.public_datasets_root_for_datasite(datasite)
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def _with_segment(self, root: Path, protocol_version: str, name: str) -> Path:
        segment = protocol_dir_name(protocol_version)
        return (root / segment / name) if segment else (root / name)

    def private_dir_for_my_dataset(
        self,
        dataset_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> Path:
        return self._with_segment(
            self.private_datasets_root(), protocol_version, dataset_name
        )

    def get_my_mock_dataset_dir(
        self,
        dataset_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> Path:
        return self.get_mock_dataset_dir(
            dataset_name=dataset_name,
            datasite=self.email,
            protocol_version=protocol_version,
        )

    def get_mock_dataset_dir(
        self,
        dataset_name: str,
        datasite: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> Path:
        return self._with_segment(
            self.public_datasets_root_for_datasite(datasite),
            protocol_version,
            dataset_name,
        )

    def get_mock_url_for_my_dataset(
        self,
        dataset_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> SyftBoxURL:
        return SyftBoxURL.from_path(
            path=self.get_my_mock_dataset_dir(dataset_name, protocol_version),
            syftbox_folder=self.syftbox_folder,
        )

    def get_private_url_for_my_dataset(
        self,
        dataset_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> SyftBoxURL:
        return SyftBoxURL.from_path(
            path=self.private_dir_for_my_dataset(dataset_name, protocol_version),
            syftbox_folder=self.syftbox_folder,
        )

    def get_readme_url_for_my_dataset(
        self,
        dataset_name: str,
        readme_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> SyftBoxURL:
        return SyftBoxURL.from_path(
            path=self.get_my_mock_dataset_dir(dataset_name, protocol_version)
            / readme_name,
            syftbox_folder=self.syftbox_folder,
        )

    def public_metadata_filename_for_my_dataset(
        self,
        dataset_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> Path:
        # TODO: not sure why the absolute is needed here
        return (
            self.get_my_mock_dataset_dir(dataset_name, protocol_version)
            / METADATA_FILENAME
        ).absolute()

    def private_metadata_filename_for_my_dataset(
        self,
        dataset_name: str,
        protocol_version: str = DATASET_PROTOCOL_VERSION,
    ) -> Path:
        return (
            self.private_dir_for_my_dataset(dataset_name, protocol_version)
            / PRIVATE_METADATA_FILENAME
        )
