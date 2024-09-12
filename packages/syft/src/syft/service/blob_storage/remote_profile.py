# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.db.stash import ObjectStash
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ..service import AbstractService


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
    pass


@serializable(canonical_name="RemoteProfileService", version=1)
class RemoteProfileService(AbstractService):
    stash: RemoteProfileStash

    def __init__(self, store: DBManager) -> None:
        self.stash = RemoteProfileStash(store=store)
