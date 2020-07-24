from ..abstract.service import NodeService
from ...message.remote_node_message import RegisterNodeMessage, RegisterNodeMessageReply


class RemoteNodeService(NodeService):
    @staticmethod
    def process(
        node: Node, msg: RegisterNodeMessage
    ) -> RegisterNodeMessageReply:
        # build a new route
        node.remote_nodes.register_node(type = msg.type, id = msg.node_id, route = msg.route)
        return RegisterNodeMessageReply(route=route, msg_id=self.id.value)

    @staticmethod
    def message_handler_types() -> List[type]:
        return [RegisterNodeMessage]
