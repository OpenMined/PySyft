from .syft_message import SyftMessage


class SaveObjectMessage(SyftMessage):
    def __init__(self, id, obj):
        self.id = id
        self.obj = obj
