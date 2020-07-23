from ...common.message import AbstractMessage
from ...common.id import UID
from ...io import Route
from ...common.token import Token

class SyftMessage(AbstractMessage):
    def __init__(self, token: Token = None, msg_id: UID = None) -> None:
        self.token = token
        self.msg_id = msg_id

class SignedMessage(SyftMessage):
    def sign(self, my_route):
        self.my_route = my_route

class SyftMessageWithReply(SignedMessage):
    def __init__(self, reply_to: Route, token: Token = None, msg_id: UID = None) -> None:
        super(self).__init__(route, msg_id)
        msg.reply_to = reply_to
