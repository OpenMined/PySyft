from .syft_message import SyftMessage


class RunFunctionOrConstructorMessage(SyftMessage):
    def __init__(self, path, args, kwargs):
        self.path = path
        self.args = args
        self.kwargs = kwargs
