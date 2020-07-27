from ...abstract.node import AbstractNode
from .common import ActionWithoutReply


class RunClassMethodAction(ActionWithoutReply):
    def __init__(self, path, _self, args, kwargs, id_at_location, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.path = path
        self._self = _self
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location

    def execute_action(self, node:AbstractNode):
        print(self.path)