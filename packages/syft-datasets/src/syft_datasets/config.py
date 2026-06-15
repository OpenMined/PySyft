from pydantic import BaseModel, Field
from pathlib import Path
from .url import SyftBoxURL

SYFT_DATASETS_FOLDER_NAME = "syft_datasets"
METADATA_FILENAME = "dataset.yaml"


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

    def public_datasets_dir_for_datasite(self, datasite: str) -> Path:
        dir = self.syftbox_folder / datasite / "public" / SYFT_DATASETS_FOLDER_NAME
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def private_dir_for_my_dataset(self, dataset_name: str) -> Path:
        return self.private_dir / SYFT_DATASETS_FOLDER_NAME / dataset_name

    def get_my_mock_dataset_dir(self, dataset_name: str) -> Path:
        return self.public_datasets_dir_for_datasite(self.email) / dataset_name

    def get_mock_dataset_dir(self, dataset_name: str, datasite: str) -> Path:
        return self.public_datasets_dir_for_datasite(datasite) / dataset_name

    def get_mock_url_for_my_dataset(self, dataset_name: str) -> SyftBoxURL:
        return SyftBoxURL.from_path(
            path=self.get_my_mock_dataset_dir(dataset_name),
            syftbox_folder=self.syftbox_folder,
        )

    def get_private_url_for_my_dataset(self, dataset_name: str) -> SyftBoxURL:
        return SyftBoxURL.from_path(
            path=self.private_dir_for_my_dataset(dataset_name),
            syftbox_folder=self.syftbox_folder,
        )

    def get_readme_url_for_my_dataset(
        self, dataset_name: str, readme_name: str
    ) -> SyftBoxURL:
        return SyftBoxURL.from_path(
            path=self.get_my_mock_dataset_dir(dataset_name) / readme_name,
            syftbox_folder=self.syftbox_folder,
        )

    def public_metadata_filename_for_my_dataset(self, dataset_name: str) -> str:
        # TODO: not sure why the absolute is needed here
        return (
            self.get_my_mock_dataset_dir(dataset_name) / METADATA_FILENAME
        ).absolute()

    def private_metadata_filename_for_my_dataset(self, dataset_name: str) -> str:
        return self.private_dir_for_my_dataset(dataset_name) / "private_metadata.yaml"
