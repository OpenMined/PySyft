# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject


@serializable()
class RemoteProfile(SyftObject):
    __canonical_name__ = "RemoteConfig"
    __version__ = SYFT_OBJECT_VERSION_2


@serializable()
class AzureRemoteProfile(RemoteProfile):
    __canonical_name__ = "AzureRemoteConfig"
    __version__ = SYFT_OBJECT_VERSION_2

    profile_name: str  # used by seaweedfs
    account_name: str
    account_key: str
    container_name: str


@serializable()
class RemoteProfileStash(BaseStash):
    object_type = RemoteProfile
    settings: PartitionSettings = PartitionSettings(
        name=RemoteProfile.__canonical_name__, object_type=RemoteProfile
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
