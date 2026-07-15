"""Backwards-compatible re-exports.

The dataset models now live under ``syft_datasets.models`` as versioned
``MigratableObject``s. This module keeps the historical import paths
(``from syft_datasets.dataset import Dataset, PrivateDatasetConfig``) working.
"""

from .models import Dataset, DatasetV1, PrivateDatasetConfig, PrivateDatasetConfigV1

__all__ = [
    "Dataset",
    "DatasetV1",
    "PrivateDatasetConfig",
    "PrivateDatasetConfigV1",
]
