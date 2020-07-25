from .syft_message import SyftMessageWithoutReply


class DeleteObjectMessage(SyftMessageWithoutReply):
    def __init__(self, obj_id, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id
