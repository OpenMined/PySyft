from syft_message import SyftMessageWithReply
from ..io.route import Route
from ...common.id import UID


class RegisterNodeMessage(RemoteNodeMessage):
    """
    This can be used to make a node aware of another.
    could potentially send keys here, eg. for a domain
    to sign up to a network.
    """
    def __init__(self, node_type: str, node_name: str, route: Route, reply_to: route,
    msg_id: UID = None):
        super().__init__(route=route, msg_id=msg_id)
        self.node_name = node_name
        self.node_type = node_type

class RegisterNodeMessageReply(RemoteNodeMessage):
    # node id etc.
    pass
