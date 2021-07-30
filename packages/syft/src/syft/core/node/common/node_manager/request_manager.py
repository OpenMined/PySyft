# stdlib
from datetime import datetime
from typing import Any
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine

# syft absolute
from syft.core.common.uid import UID
from syft.core.node.common.node_service.request_receiver.request_receiver_messages import (
    RequestStatus,
)

# relative
from ..exceptions import RequestError
from ..node_table.request import Request
from .database_manager import DatabaseManager


class RequestManager(DatabaseManager):

    schema = Request

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=RequestManager.schema, db=database)

    def first(self, **kwargs: Any) -> Request:
        result = super().first(**kwargs)
        if not result:
            raise RequestError

        return result

    def create_request(
        self,
        user_id: int,
        user_name: str,
        object_id: str,
        reason: str,
        request_type: str,
        verify_key: Optional[VerifyKey] = None,
        tags: Optional[List[str]] = None,
        object_type: str = "",
    ) -> None:
        date = datetime.now()

        return self.register(
            id=str(UID().value),
            user_id=user_id,
            user_name=user_name,
            object_id=object_id,
            date=date,
            reason=reason,
            request_type=request_type,
            verify_key=verify_key,
            tags=tags,
            object_type=object_type,
        )

    def status(self, request_id: int) -> RequestStatus:
        _req = self.first(id=request_id)
        if _req.status == "pending":
            return RequestStatus.Pending
        elif _req.status == "accepted":
            return RequestStatus.Accepted
        else:
            return RequestStatus.Rejected

    def set(self, request_id: int, status: RequestStatus) -> None:
        self.modify({"id": request_id}, {"status": status})
