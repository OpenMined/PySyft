# relative
from ...serde.serializable import serializable
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionSettings
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject


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


@serializable(canonical_name="RemoteProfileSQLStash", version=1)
class RemoteProfileStash(ObjectStash[RemoteProfile]):
    object_type = RemoteProfile
    settings: PartitionSettings = PartitionSettings(
        name=RemoteProfile.__canonical_name__, object_type=RemoteProfile
    )
