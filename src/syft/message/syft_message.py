from ...common.message import AbstractMessage
from ...common.id import UID
from ...io import Route

class SyftMessage(AbstractMessage):
    def __init__(self, route: Route, msg_id: UID = None) -> None:
        self.route = route
        self.msg_id = msg_id

class SyftMessageWithReply(SyftMessage):
    def __init__(self, route: Route, reply_to: Route, UID = None) -> None:
        msg = super(self, route, UID)
        msg.reply_to = reply_to
