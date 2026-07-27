from __future__ import annotations

from pathlib import Path
from uuid import UUID

from syft_migration import MigratableObject
from syft_notebook_ui.formatter_mixin import PydanticFormatterMixin

from ...migrations import dataset_registry


class PrivateDatasetConfigV1(
    MigratableObject, PydanticFormatterMixin, registry=dataset_registry
):
    """Private dataset metadata, stored as private_metadata.yaml outside the sync folder."""

    canonical_name: str = "PrivateDatasetConfig"
    version: str = "1"

    uid: UUID  # id for this dataset
    data_dir: Path

    def disk_dict(self) -> dict:
        return self.model_dump(mode="json")
