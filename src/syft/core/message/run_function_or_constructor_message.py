from .syft_message import SyftMessage


class RunFunctionOrConstructorMessage(SyftMessage):
    def __init__(self, path, args, kwargs, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)
        self.path = path
        self.args = args
        self.kwargs = kwargs
