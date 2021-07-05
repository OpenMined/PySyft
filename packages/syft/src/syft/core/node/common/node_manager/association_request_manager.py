# stdlib
from datetime import datetime
from typing import List
from typing import Union

# relative
from ..exceptions import AssociationRequestError
from ..node_table.association import Association
from ..node_table.association_request import AssociationRequest
from .database_manager import DatabaseManager


class AssociationRequestManager(DatabaseManager):

    schema = AssociationRequest

    def __init__(self, database):
        self._schema = AssociationRequestManager.schema
        self.db = database

    def first(self, **kwargs) -> Union[None, List]:
        result = super().first(**kwargs)
        if not result:
            raise AssociationRequestError

        return result

    def create_association_request(self, node, **kwargs):
        date = datetime.now()
        if super().first(node=node):
            raise Exception("Association request name already exists!")

        kwargs["node"] = node
        kwargs["requested_date"] = datetime.now().strftime("%m/%d/%Y")
        self.register(**kwargs)

    def associations(self):
        return list(self.db.session.query(Association).all())

    def association(self, **kwargs):
        return self.db.session.query(Association).filter_by(**kwargs).first()

    def set(self, node_name, response):
        self.modify(
            {"node": node_name},
            {"status": response, "accepted_date": datetime.now().strftime("%m/%d/%Y")},
        )
