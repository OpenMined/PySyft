# stdlib
from datetime import datetime
from typing import Any

# third party
from sqlalchemy.engine import Engine

# relative
from ...enums import RequestAPIFields
from ..exceptions import AssociationRequestError
from ..node_table.association_request import AssociationRequest
from .database_manager import DatabaseManager


class AssociationRequestManager(DatabaseManager):

    schema = AssociationRequest

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=AssociationRequestManager.schema, db=database)

    def first(self, **kwargs: Any) -> AssociationRequest:
        result = super().first(**kwargs)
        if not result:
            raise AssociationRequestError

        return result

    def create_association_request(
        self,
        node_name: str,
        node_address: str,
        node_id: str,
        status: str,
        name: str = "",
        email: str = "",
        reason: str = "",
    ) -> None:
        table_fields = {}

        table_fields[RequestAPIFields.NODE_NAME] = node_name
        table_fields[RequestAPIFields.NODE_ID] = node_id
        table_fields[RequestAPIFields.REQUESTED_DATE] = datetime.now().strftime(
            "%m/%d/%Y"
        )
        table_fields[RequestAPIFields.STATUS] = status
        table_fields[RequestAPIFields.NODE_ADDRESS] = node_address
        table_fields[RequestAPIFields.NODE_ADDRESS] = node_address
        table_fields[RequestAPIFields.NODE_ADDRESS] = node_address
        table_fields[RequestAPIFields.NODE_ADDRESS] = node_address
        table_fields[RequestAPIFields.NAME] = name
        table_fields[RequestAPIFields.EMAIL] = email
        table_fields[RequestAPIFields.REASON] = reason

        try:
            self.first(**{RequestAPIFields.NODE_ADDRESS.value: node_address})
            self.modify(
                query={RequestAPIFields.NODE_ADDRESS.value: node_address},
                values=table_fields,
            )
        except AssociationRequestError:
            # no existing AssociationRequest so lets make one
            self.register(**table_fields)  # type: ignore

    def set(self, node_address: str, response: str) -> None:
        self.modify(
            query={RequestAPIFields.NODE_ADDRESS.value: node_address},
            values={
                RequestAPIFields.STATUS.value: response,
                RequestAPIFields.ACCEPTED_DATE.value: datetime.now().strftime(
                    "%m/%d/%Y"
                ),
            },
        )
