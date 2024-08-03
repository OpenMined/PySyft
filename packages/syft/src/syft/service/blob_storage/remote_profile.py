# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash, DocumentStore, PartitionSettings
from ...types.syft_object import SYFT_OBJECT_VERSION_1, SyftObject


@serializable()
class RemoteProfile(SyftObject):
    __canonical_name__ = "RemoteConfig"
    __version__ = SYFT_OBJECT_VERSION_1


@serializable()
class AzureRemoteProfile(RemoteProfile):
    __canonical_name__ = "AzureRemoteConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    profile_name: str  # used by seaweedfs
    account_name: str
    account_key: str
    container_name: str


@serializable(canonical_name="RemoteProfileStash", version=1)
class RemoteProfileStash(BaseUIDStoreStash):
    object_type = RemoteProfile
    settings: PartitionSettings = PartitionSettings(
        name=RemoteProfile.__canonical_name__, object_type=RemoteProfile,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
