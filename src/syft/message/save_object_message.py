from .syft_message import SyftMessage
from ...common.token import Token

class SaveObjectMessage(SyftMessage):
    def __init__(self, id, obj, token=None, msg_id=None):
        super().__init__(token=token, msg_id=msg_id)
        self.id = id
        self.obj = obj
