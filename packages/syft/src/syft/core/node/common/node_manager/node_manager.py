# stdlib
from typing import Optional

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine

# relative
from ..node_service.node_credential.node_credentials import NodeCredentials
from ..node_table.node import Node
from .database_manager import DatabaseManager


class NodeManager(DatabaseManager):
    schema = Node

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=NodeManager.schema, db=database)

    def create_or_get_node(self, node_uid: str, node_name: str) -> int:
        # node_uid is a UID as a string with no_dash
        node = self.first(node_uid=node_uid)
        if node:
            self.modify(query={"node_uid": node_uid}, values={"node_name": node_name})
        else:
            self.register(**{"node_uid": node_uid, "node_name": node_name})
            node = self.first(node_uid=node_uid)
        return node.id

    def add_or_update_node_credentials(self, credentials: NodeCredentials) -> None:
        node = self.first(node_uid=credentials.node_uid)
        if node:
            if node.verify_key is not None:
                credentials.validate(key=node.verify_key)
            self.modify(
                query={"node_uid": credentials.node_uid}, values={**credentials}
            )
        else:
            self.register(**credentials)

    def validate_id_and_key(
        self, node_uid: str, verify_key: VerifyKey
    ) -> Optional[Node]:
        return self.first(
            node_uid=node_uid,
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8"),
        )

    def get_node_for(self, verify_key: VerifyKey) -> Optional[Node]:
        return self.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )
