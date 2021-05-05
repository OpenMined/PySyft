# stdlib
from datetime import datetime
from typing import List
from typing import Union

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

    def create_request(self, user_id, user_name, object_id, reason, request_type):
        date = datetime.now()

        return self.register(
            user_id=user_id,
            user_name=user_name,
            object_id=object_id,
            date=date,
            reason=reason,
            request_type=request_type,
        )

    def set(self, request_id, status):
        self.modify({"id": request_id}, {"status": status})
