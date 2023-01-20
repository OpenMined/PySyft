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
        source: str,
        target: str,
        status: str,
    ) -> None:
        table_fields = {}
        table_fields["node_name"] = node_name
        table_fields[RequestAPIFields.REQUESTED_DATE] = datetime.now().strftime(
            "%m/%d/%Y"
        )

        table_fields[RequestAPIFields.SOURCE] = source
        table_fields[RequestAPIFields.TARGET] = target
        table_fields[RequestAPIFields.STATUS] = status
        table_fields["node_address"] = node_address

        try:
            self.first(**{"source": source, "target": target})
            self.modify(query={"source": source, "target": target}, values=table_fields)
        except AssociationRequestError:
            # no existing AssociationRequest so lets make one
            self.register(**table_fields)  # type: ignore

    def set(self, source: str, target: str, response: str) -> None:
        self.modify(
            query={"source": source, "target": target},
            values={
                "status": response,
                "accepted_date": datetime.now().strftime("%m/%d/%Y"),
            },
        )
