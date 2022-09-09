# stdlib
from datetime import datetime
from typing import Any
from typing import Dict

# relative
from ...enums import RequestAPIFields
from ..exceptions import AssociationRequestError
from ..node_table.association_request import NoSQLAssociationRequest
from .database_manager import NoSQLDatabaseManager


class NoSQLAssociationRequestManager(NoSQLDatabaseManager):
    """Class to manage user database actions."""

    _collection_name = "association_requests"
    __canonical_object_name__ = "AssociationRequest"

    def first(self, **kwargs: Any) -> NoSQLAssociationRequest:
        result = super().find_one(kwargs)
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
            self.update(
                search_params={"source": source, "target": target},
                updated_args=table_fields,
            )
        except AssociationRequestError:
            # no existing AssociationRequest so lets make one
            curr_len = len(self)
            association_request = NoSQLAssociationRequest(
                id_int=curr_len + 1,
                **table_fields,
            )
            self.add(association_request)

    # modify in association field before testing.
    def accept_or_deny(self, source: str, target: str, response: str) -> None:
        association_request = self.first(**{"source": source, "target": target})
        association_request.status = response
        association_request.processed_date = str(datetime.now().strftime("%m/%d/%Y"))
        attributes: Dict[str, Any] = {}
        attributes["__blob__"] = association_request.to_bytes()

        self.update_one(
            query={"source": source, "target": target},
            values=attributes,
        )
