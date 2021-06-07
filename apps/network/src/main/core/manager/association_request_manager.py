# stdlib
from datetime import datetime
from hashlib import sha256
from typing import List
from typing import Union

# grid relative
from ..database.association.association import Association
from ..database.association.request import AssociationRequest
from ..exceptions import AssociationRequestError
from .database_manager import DatabaseManager
from .role_manager import RoleManager


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

    def create_association_request(self, name, address, sender_address):
        date = datetime.now()
        if super().first(name=name):
            raise Exception("Association request name already exists!")

        handshake_value = self.__generate_hash(name)

        return self.register(
            date=date,
            name=name,
            address=address,
            sender_address=sender_address,
            handshake_value=handshake_value,
        )

    def associations(self):
        return list(self.db.session.query(Association).all())

    def association(self, **kwargs):
        return self.db.session.query(Association).filter_by(**kwargs).first()

    def set(self, handshake, value):
        accepted_value = value == "accept"
        if accepted_value:
            req = self.first(handshake_value=handshake)
            new_association = Association(
                name=req.name, address=req.address, date=datetime.now()
            )
            self.db.session.add(new_association)
        self.modify(
            {"handshake_value": handshake},
            {"pending": False, "accepted": accepted_value},
        )

    def __generate_hash(self, name):
        initial_string = name
        initial_string_encoded = initial_string.encode("UTF-8")
        hashed = sha256(initial_string_encoded)
        hashed = hashed.hexdigest()

        return hashed
