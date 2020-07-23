from .syft_message import SyftMessage
from ...common.token import Token

class DeleteObjectMessage(SyftMessage):
    def __init__(self, obj_id, token=None, msg_id=None):
        super().__init__(token=token, msg_id=msg_id)
        self.obj_id = obj_id
