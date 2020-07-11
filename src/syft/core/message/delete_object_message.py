from .syft_message import SyftMessage


class DeleteObjectMessage(SyftMessage):
    def __init__(self, id):
        self.id = id
