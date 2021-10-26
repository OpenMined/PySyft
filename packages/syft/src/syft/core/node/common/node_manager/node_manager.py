# third party
from sqlalchemy.engine import Engine

# relative
from ..node_table.node import Node
from .database_manager import DatabaseManager


class NodeManager(DatabaseManager):
    schema = Node

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=NodeManager.schema, db=database)

    def create_or_get_node(self, node_uid: str, node_name: str) -> Node:
        print("1", node_uid, node_name)
        # node_uid is a UID as a string with no_dash
        node = self.first(node_uid=node_uid)
        print("2")
        print("is there a node with this id", node, node_uid)
        if node:
            print("update the node")
            self.modify(query={"node_uid": node_uid}, values={"node_name": node_name})
        else:
            self.register(**{"node_uid": node_uid, "node_name": node_name})
            print(
                "we just created the node whats the obj",
                type(node),
                node.id,  # type: ignore
                node.node_uid,  # type: ignore
            )
            node = self.first(node_uid=node_uid)
        return node
