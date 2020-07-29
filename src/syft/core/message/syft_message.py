from syft.interfaces.message import AbstractMessage


class SyftMessage(AbstractMessage):
    def __init__(self, route, msg_id=None):
        self.route = route
        self.msg_id = msg_id
