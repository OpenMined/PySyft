from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

import yaml
from pydantic import Field
from syft_migration import MigratableObject
from syft_notebook_ui.formatter_mixin import (
    ANSIPydanticFormatter,
    PydanticFormatter,
    PydanticFormatterMixin,
)
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from ...config import (
    PRIVATE_METADATA_FILENAME,
    SYFT_DATASETS_FOLDER_NAME,
    SyftBoxConfig,
    protocol_dir_name,
)
from ...migrations import dataset_registry
from ...migrations.registry import DATASET_PROTOCOL_VERSION
from ...url import SyftBoxURL
from ..private_dataset_config.v1 import PrivateDatasetConfigV1


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class DatasetV1(MigratableObject, PydanticFormatterMixin, registry=dataset_registry):
    """Public dataset metadata, stored as dataset.yaml under
    SyftBox/<datasite>/public/syft_datasets/[v<n>/]<name>/."""

    __display_formatter__: ClassVar[PydanticFormatter] = ANSIPydanticFormatter()
    __table_extra_fields__: ClassVar[list[str]] = ["name", "owner"]

    canonical_name: str = "Dataset"
    version: str = "1"

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

    # Runtime-only: set by DatasetStorage / the manager, never serialized. The
    # on-disk protocol layout the dataset lives in (governs the v<n> segment).
    _syftbox_config: SyftBoxConfig | None = None
    _protocol_version: str = DATASET_PROTOCOL_VERSION

    def disk_dict(self) -> dict:
        """The on-disk form of the dataset metadata."""
        return self.model_dump(mode="json")

    @property
    def owner(self) -> str:
        return self.mock_url.host

    @property
    def syftbox_config(self) -> SyftBoxConfig:
        if self._syftbox_config is None:
            raise ValueError("SyftBox config is not set.")
        return self._syftbox_config

    def _url_to_path(self, url: SyftBoxURL) -> Path:
        return url.to_local_path(syftbox_folder=self.syftbox_config.syftbox_folder)

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
        return self._private_metadata_dir / PRIVATE_METADATA_FILENAME

    @cached_property
    def private_config(self) -> PrivateDatasetConfigV1:
        config_path = self.private_config_path
        if not config_path.exists():
            raise FileNotFoundError(
                f"Private dataset config not found at {config_path}"
            )
        data = yaml.safe_load(config_path.read_text()) or {}
        data.setdefault("canonical_name", "PrivateDatasetConfig")
        data.setdefault("version", "1")
        return PrivateDatasetConfigV1(**data)

    @property
    def private_dir(self) -> Path:
        """The private data dir for this dataset, under the dataset's protocol layout.

        Derived from the path (owner + name + protocol) rather than the stored
        URL so it stays correct across on-disk layouts.
        """
        segment = protocol_dir_name(self._protocol_version)
        root = (
            self.syftbox_config.syftbox_folder
            / self.owner
            / "private"
            / SYFT_DATASETS_FOLDER_NAME
        )
        return (root / segment / self.name) if segment else (root / self.name)

    @property
    def _private_metadata_dir(self) -> Path:
        if self.syftbox_config.email != self.owner:
            raise ValueError(
                "Cannot access private data for a dataset owned by another user."
            )
        return self.private_dir

    @property
    def mock_files(self) -> list[Path]:
        """Absolute paths to all mock files uploaded during dataset.create.

        Excludes dataset.yaml and readme.md files.
        """
        return [self._url_to_path(url) for url in self.mock_files_urls]

    @property
    def private_files(self) -> list[Path]:
        """Absolute paths to all private files.

        For owners: returns paths from dataset.create (private_files_paths).
        For non-owners (e.g. enclave): returns files from shared_private_dir.
        """
        return [
            f
            for f in self.private_dir.iterdir()
            if f.is_file()
            and f.name not in (PERMISSION_FILE_NAME, PRIVATE_METADATA_FILENAME)
        ]

    @property
    def files(self) -> list[Path]:
        """Absolute paths to all files (both mock and private)."""
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
