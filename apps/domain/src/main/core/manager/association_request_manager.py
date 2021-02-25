from typing import Union
from typing import List
from hashlib import sha256
from datetime import datetime

from .database_manager import DatabaseManager
from ..database.association.request import AssociationRequest
from .role_manager import RoleManager
from ..exceptions import AssociationRequestError


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

    def create_association_request(self, name, address):
        date = datetime.now()
        handshake_value = self.__generate_hash(name)

        return self.register(
            date=date,
            name=name,
            address=address,
            handshake_value=handshake_value,
        )

    def __generate_hash(self, name):
        initial_string = name
        initial_string_encoded = initial_string.encode("UTF-8")
        hashed = sha256(initial_string_encoded)
        hashed = hashed.hexdigest()

        return hashed

    def set(self, handshake, value):
        accepeted_value = value == "accept"

        self.modify(
            {"handshake_value": handshake},
            {"pending": False, "accepted": accepeted_value},
        )