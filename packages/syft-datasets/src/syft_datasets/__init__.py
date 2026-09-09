# __version__ comes from the installed distribution metadata (see version.py).
from .version import __version__

from .config import SyftBoxConfig
from .dataset_ref import DatasetRef
from .dataset_storage import DatasetStorage
from .migrations import dataset_registry
from .migrations.history import register_historic_schemas
from .models import Dataset, DatasetV1, PrivateDatasetConfig, PrivateDatasetConfigV1

# Historic schemas list object versions that must already be registered, which
# happens when the models above are imported.
register_historic_schemas()

__all__ = [
    "__version__",
    "SyftBoxConfig",
    "DatasetRef",
    "DatasetStorage",
    "dataset_registry",
    # Models
    "Dataset",
    "DatasetV1",
    "PrivateDatasetConfig",
    "PrivateDatasetConfigV1",
]
