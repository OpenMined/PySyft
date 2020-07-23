from .syft_message import SyftMessage


class RunFunctionOrConstructorMessage(SyftMessage):
    def __init__(self, path, args, kwargs, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.path = path
        self.args = args
        self.kwargs = kwargs
