"""Shared mock objects for the syft-dataset migration tests."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from syft_datasets.models import DatasetV1, PrivateDatasetConfigV1
from syft_datasets.url import SyftBoxURL

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
DATASET_NAME = "demo"
DATASET_UID = UUID("00000000-0000-0000-0000-000000000001")


def create_mock_dataset() -> DatasetV1:
    base = f"syft://{DO_EMAIL}/public/syft_datasets/{DATASET_NAME}"
    return DatasetV1(
        uid=DATASET_UID,
        created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        name=DATASET_NAME,
        summary="demo dataset",
        tags=["demo"],
        mock_url=SyftBoxURL(base),
        private_url=SyftBoxURL(
            f"syft://{DO_EMAIL}/private/syft_datasets/{DATASET_NAME}"
        ),
        readme_url=SyftBoxURL(f"{base}/readme.md"),
        mock_files_urls=[SyftBoxURL(f"{base}/mock.csv")],
    )


def create_mock_private_config() -> PrivateDatasetConfigV1:
    return PrivateDatasetConfigV1(uid=DATASET_UID, data_dir=Path(""))
