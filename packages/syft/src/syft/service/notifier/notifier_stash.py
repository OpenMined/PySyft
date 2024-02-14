# stdlib
from typing import List

# third party
from result import Ok
from result import Result
from result import Err

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from ..response import SyftError
from .notifier import NotifierSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=List[UID])


@instrument
@serializable()
class NotifierStash(BaseStash):
    object_type = NotifierSettings
    settings: PartitionSettings = PartitionSettings(
        name=NotifierSettings.__canonical_name__, object_type=NotifierSettings
    )

    # Usual Stash Implementation
    # The only diff is: We need to be sure that
    # this will act as a Singleton (as SettingsStash).

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get(self, credentials: SyftVerifyKey) -> Result[NotifierSettings, Err]:
        """Get Settings"""
        result = self.get_all(credentials)
        if result.is_ok():
            settings = result.ok()
            if len(settings) == 0:
                return Ok(None)
            result = settings[0]
            return Ok(result)
        else:
            return Err(message=result.err())

    def set(
        self, credentials: SyftVerifyKey, settings: NotifierSettings
    ) -> Result[NotifierSettings, str]:
        res = self.check_type(settings, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(credentials=credentials, obj=res.ok())

    def update(
        self, credentials: SyftVerifyKey, settings: NotifierSettings
    ) -> Result[NotifierSettings, str]:
        res = self.check_type(settings, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().update(credentials=credentials, obj=res.ok())
