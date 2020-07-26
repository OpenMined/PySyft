from .syft_message import SyftMessageWithReply
from .syft_message import SyftMessage
from ..io.address import Address
from ...common.id import UID


class RegisterNodeMessage(SyftMessageWithReply):
    """
    This can be used to make a node aware of another.
    could potentially send keys here, eg. for a domain
    to sign up to a network.
    """

    def __init__(
        self,
        node_type: str,
        node_name: str,
        reply_to: Address,
        address: Address,
        msg_id: UID = None,
    ):
        super().__init__(address=address, reply_to=reply_to, msg_id=msg_id)
        self.node_name = node_name
        self.node_type = node_type


class RegisterNodeMessageReply(SyftMessage):
    # node id etc.
    pass
