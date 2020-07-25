from .syft_message import SyftMessage

class SaveObjectMessage(SyftMessage):
    def __init__(self, obj_id, obj, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id
        self.obj = obj
