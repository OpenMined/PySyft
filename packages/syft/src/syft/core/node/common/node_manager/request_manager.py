# stdlib
from datetime import datetime
from typing import Any
from typing import Dict

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ....common.uid import UID
from ..exceptions import RequestError
from ..node_service.request_receiver.request_receiver_messages import RequestStatus
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

    def create_request(self, **kwargs) -> Request:
        date = datetime.now()
        return self.register(id=str(UID().value), date=date, **kwargs)

    def status(self, request_id: str) -> RequestStatus:
        _req = self.first(id=request_id)
        if _req.status == "pending":
            return RequestStatus.Pending
        elif _req.status == "accepted":
            return RequestStatus.Accepted
        else:
            return RequestStatus.Rejected

    def set(self, request_id: int, status: RequestStatus) -> None:
        self.modify({"id": request_id}, {"status": status})

    def get_user_info(request_id: int) -> Dict:
        request = super().first(id=request_id)
        return {
            "name": request.user_name,
            "email": request.user_email,
            "role": request.user_role,
            "budget": request.user_budget,
            "institution": request.institution,
            "website": request.website,
        }

    def clear(self) -> None:
        local_session = sessionmaker(bind=self.db)()
        local_session.query(self.schema).delete()
        local_session.commit()
        local_session.close()
