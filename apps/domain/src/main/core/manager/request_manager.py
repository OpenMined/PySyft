# stdlib
from datetime import datetime
from typing import List
from typing import Union

# third party
from syft.core.common.uid import UID
from syft.core.node.domain.service import RequestStatus

# grid relative
from ..database.requests.request import Request
from ..exceptions import RequestError
from .database_manager import DatabaseManager
from .role_manager import RoleManager


class RequestManager(DatabaseManager):

    schema = Request

    def __init__(self, database):
        self._schema = RequestManager.schema
        self.db = database

    def first(self, **kwargs) -> Union[None, List]:
        result = super().first(**kwargs)
        if not result:
            raise RequestError

        return result

    def create_request(
        self,
        user_id,
        user_name,
        object_id,
        reason,
        request_type,
        verify_key=None,
        tags=[],
        object_type="",
    ):
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

    def status(self, request_id):
        _req = self.first(id=request_id)
        if _req.status == "pending":
            return RequestStatus.pending
        elif _req.status == "accepted":
            return RequestStatus.Accepted
        else:
            return RequestStatus.Rejected

    def set(self, request_id, status):
        self.modify({"id": request_id}, {"status": status})
