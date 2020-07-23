from .syft_message import SyftMessage
from ...common.token import Token

class RunFunctionOrConstructorMessage(SyftMessage):
    def __init__(self, path, args, kwargs, token=None, msg_id=None):
        super().__init__(token=token, msg_id=msg_id)
        self.path = path
        self.args = args
        self.kwargs = kwargs
