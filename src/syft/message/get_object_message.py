from .syft_message import SyftMessage


class GetObjectMessage(SyftMessage):
    def __init__(self, obj_id, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)
        self.obj_id = obj_id
