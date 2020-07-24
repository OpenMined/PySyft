from ...common.message import AbstractMessage
from ...common.id import UID
from ..io.route import Route
from ..io.address import Address

class SyftMessage(AbstractMessage):
    def __init__(self, address: Address, msg_id: UID = None) -> None:
        self.msg_id = msg_id

class SignedMessage(SyftMessage):
    def sign(self, my_route):
        self.my_route = my_route

class SyftMessageWithReply(SignedMessage):
    def __init__(self, reply_to: Address,  address: Address, msg_id: UID = None) -> None:
        super(self).__init__(route, msg_id)
        msg.reply_to = reply_to
