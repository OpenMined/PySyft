from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import ClassVar
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME
from typing_extensions import Self
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, Field
from syft_notebook_ui.formatter_mixin import (
    ANSIPydanticFormatter,
    PydanticFormatter,
    PydanticFormatterMixin,
)

from .types import PathLike, to_path
from .url import SyftBoxURL
from .config import SyftBoxConfig


def _utcnow():
    return datetime.now(tz=timezone.utc)


class DatasetBase(BaseModel):
    __display_formatter__: ClassVar[PydanticFormatter] = ANSIPydanticFormatter()
    _syftbox_config: SyftBoxConfig | None = None

    def save(self, filepath: PathLike) -> None:
        filepath = to_path(filepath)
        if not filepath.suffix == ".yaml":
            raise ValueError("Model must be saved as a .yaml file.")

        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        yaml_dump = yaml.safe_dump(data, indent=2, sort_keys=False)
        filepath.write_text(yaml_dump)

    @classmethod
    def load(
        cls, filepath: PathLike, syftbox_config: SyftBoxConfig | None = None
    ) -> Self:
        filepath = to_path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        data = yaml.safe_load(filepath.read_text())
        res = cls.model_validate(data)
        res._syftbox_config = syftbox_config
        return res

    def __str__(self) -> str:
        return self.__display_formatter__.format_str(self)

    def __repr__(self) -> str:
        return self.__display_formatter__.format_repr(self)

    def _repr_html_(self) -> str:
        return self.__display_formatter__.format_html(self)

    def _repr_markdown_(self) -> str:
        return self.__display_formatter__.format_markdown(self)


class PrivateDatasetConfig(DatasetBase, PydanticFormatterMixin):
    """Used to store private dataset metadata, outside of the sync folder."""

    uid: UUID  # id for this dataset
    data_dir: Path


class Dataset(DatasetBase, PydanticFormatterMixin):
    __table_extra_fields__ = [
        "name",
        "owner",
    ]

    uid: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    name: str
    summary: str | None = None
    tags: list[str] = []
    location: str | None = None

    mock_url: SyftBoxURL
    private_url: SyftBoxURL
    readme_url: SyftBoxURL | None = None

    # URLs to uploaded files (excluding metadata files)
    mock_files_urls: list[SyftBoxURL] = Field(default_factory=list)

    @property
    def owner(self) -> str:
        return self.mock_url.host

    @property
    def syftbox_config(self) -> SyftBoxConfig:
        if self._syftbox_config is None:
            raise ValueError("SyftBox config is not set.")
        return self._syftbox_config

    def _url_to_path(self, url: SyftBoxURL) -> Path:
        return url.to_local_path(
            syftbox_folder=self.syftbox_config.syftbox_folder,
        )

    @property
    def readme_path(self) -> Path | None:
        if self.readme_url is None:
            return None
        return self._url_to_path(self.readme_url)

    def get_readme(self) -> str | None:
        """Get the content of the README file."""
        if self.readme_path and self.readme_path.exists():
            return self.readme_path.read_text()
        return None

    @property
    def mock_dir(self) -> Path:
        return self._url_to_path(self.mock_url)

    @property
    def private_config_path(self) -> Path:
        if self.syftbox_config.email != self.owner:
            raise ValueError(
                "Cannot access private config for a dataset owned by another user."
            )
        return self._private_metadata_dir / "private_metadata.yaml"

    @cached_property
    def private_config(self) -> PrivateDatasetConfig:
        config_path = self.private_config_path
        if not config_path.exists():
            raise FileNotFoundError(
                f"Private dataset config not found at {config_path}"
            )

        return PrivateDatasetConfig.load(
            filepath=config_path, syftbox_config=self._syftbox_config
        )

    @property
    def private_dir(self) -> Path:
        if self._syftbox_config is None:
            raise ValueError("SyftBox config is not set.")
        private_dir = (
            self.syftbox_config.syftbox_folder
            / self.owner
            / "private"
            / "syft_datasets"
            / self.name
        )
        return private_dir

    @property
    def _private_metadata_dir(self) -> Path:
        if self.syftbox_config.email != self.owner:
            raise ValueError(
                "Cannot access private data for a dataset owned by another user."
            )
        return self.private_dir

    @property
    def mock_files(self) -> list[Path]:
        """
        Get absolute paths to all mock files uploaded during dataset.create.
        Excludes dataset.yaml and readme.md files.
        """
        return [self._url_to_path(url) for url in self.mock_files_urls]

    @property
    def private_files(self) -> list[Path]:
        """
        Get absolute paths to all private files.

        For owners: returns paths from dataset.create (private_files_paths).
        For non-owners (e.g. enclave): returns files from shared_private_dir.
        """
        return [
            f
            for f in self.private_dir.iterdir()
            if f.is_file()
            and f.name not in (PERMISSION_FILE_NAME, "private_metadata.yaml")
        ]

    @property
    def files(self) -> list[Path]:
        """
        Get absolute paths to all files (both mock and private) uploaded during dataset.create.
        """
        return self.mock_files + self.private_files

    def _generate_description_html(self) -> str:
        from syft_notebook_ui.pydantic_html_repr import create_html_repr

        fields_to_include = ["name", "created_at", "summary", "tags", "location"]

        paths_to_include = []
        try:
            paths_to_include.append("mock_dir")
        except Exception:
            fields_to_include.append("mock_url")

        try:
            private_dir = self.private_dir
            if private_dir.is_dir():
                paths_to_include.append("private_dir")
        except Exception:
            pass

        try:
            readme_path = self.readme_path
            if readme_path and readme_path.exists():
                paths_to_include.append("readme_path")
        except Exception:
            fields_to_include.append("readme_url")

        description = create_html_repr(
            obj=self,
            fields=fields_to_include,
            display_paths=paths_to_include,
        )

        hint = (
            '<div style="margin-top: 8px; font-size: 0.9em; color: #666;">'
            "💡 Use <code>.mock_files</code> to access mock data"
            "</div>"
        )
        return description + hint

    def describe(self) -> None:
        from IPython.display import HTML, display

        description = self._generate_description_html()
        display(HTML(description))

    def _repr_html_(self) -> str:
        return self._generate_description_html()
