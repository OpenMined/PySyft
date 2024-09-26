# future
from __future__ import annotations

# third party
from pydantic import BaseModel
from pydantic import Field

# relative
from ..serde.serializable import serializable
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftBaseObject
from .locks import LockingConfig
from .locks import NoLockingConfig


class StoreClientConfig(BaseModel):
    """Base Client specific configuration"""

    pass


@serializable(
    attrs=["settings", "store_config", "unique_cks", "searchable_cks"],
    canonical_name="StorePartition",
    version=1,
)
class StorePartition:
    """Base StorePartition
    Parameters:
        settings: PartitionSettings
            PySyft specific settings
        store_config: StoreConfig
            Backend specific configuration
    """


@serializable()
class StoreConfig(SyftBaseObject):
    """Base Store configuration

    Parameters:
        store_type: Type
            Document Store type
        client_config: Optional[StoreClientConfig]
            Backend-specific config
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
            Defaults to NoLockingConfig.
    """

    __canonical_name__ = "StoreConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    store_type: type[DocumentStore]
    client_config: StoreClientConfig | None = None
    locking_config: LockingConfig = Field(default_factory=NoLockingConfig)  # noqa: F821


@serializable(canonical_name="DocumentStore", version=1)
class DocumentStore:
    pass
