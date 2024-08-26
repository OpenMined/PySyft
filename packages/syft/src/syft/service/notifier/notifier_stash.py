# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from .notifier import NotifierSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@instrument
@serializable(canonical_name="NotifierStashSQL", version=1)
class NotifierStash(ObjectStash[NotifierSettings]):
    object_type = NotifierSettings
    settings: PartitionSettings = PartitionSettings(
        name=NotifierSettings.__canonical_name__, object_type=NotifierSettings
    )

    # TODO: should this method behave like a singleton?
    def get(self, credentials: SyftVerifyKey) -> Result[NotifierSettings, Err]:
        """Get Settings"""
        # actually get latest settings
        results = self.get_all(credentials, limit=1)
        match results:
            case Ok(settings) if len(settings) > 0:
                return Ok(settings[0])
            case Ok(_):
                return Ok(None)
            case Err(e):
                return Err(e)
