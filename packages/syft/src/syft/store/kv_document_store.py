# relative
from .document_store import StorePartition


class KeyValueBackingStore:
    pass


class KeyValueStorePartition(StorePartition):
    """Key-Value StorePartition
    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings
        `store_config`: StoreConfig
            Backend specific configuration
    """

    pass
