from .syft_message import SyftMessage


class SaveObjectMessage(SyftMessage):
    def __init__(self, id, obj, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)
        self.id = id
        self.obj = obj
