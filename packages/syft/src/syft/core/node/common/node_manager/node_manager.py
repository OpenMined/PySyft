# stdlib
from typing import Any
from typing import Dict
from typing import List

# third party
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query

# relative
from ..exceptions import RoleNotFoundError
from ..node_table.node import Node
from .database_manager import DatabaseManager


class NodeManager(DatabaseManager):
    schema = Node

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=NodeManager.schema, db=database)

    def create_or_get_node(self, node_id: str, node_name: str) -> Node:
        # node_id is a UID as a string with no_dash
        node = self.first(node_id=node_id)
        if node:
            self.modify({"node_id": node_id}, {"node_name": node_name})
        else:
            node = self.register({"node_id": node_id, "node_name": node_name})
        return node
