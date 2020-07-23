from syft_message import SyftMessageWithReply, SyftMessage
from ..io.route import Route
from ...common.id import UID
from ...common.token import Token

class RegisterNodeMessage(SyftMessageWithReply):
    """
    This can be used to make a node aware of another.
    could potentially send keys here, eg. for a domain
    to sign up to a network.
    """
    def __init__(self, node_type: str, node_name: str, reply_to: Route,
    token: Token = None, msg_id: UID = None):
        super().__init__(token=token, reply_to=reply_to, msg_id=msg_id)
        self.node_name = node_name
        self.node_type = node_type

class RegisterNodeMessageReply(SyftMessage):
    # node id etc.
    pass
