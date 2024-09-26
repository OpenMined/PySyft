# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...util.telemetry import instrument
from .notifier import NotifierSettings


@instrument
@serializable(canonical_name="NotifierSQLStash", version=1)
class NotifierStash(ObjectStash[NotifierSettings]):
    @as_result(StashException, NotFoundException)
    def get(self, credentials: SyftVerifyKey) -> NotifierSettings:
        """Get Settings"""
        # actually get latest settings
        result = self.get_all(credentials, limit=1, sort_order="desc").unwrap()
        if len(result) > 0:
            return result[0]
        raise NotFoundException(
            public_message="No settings found for the current user."
        )
