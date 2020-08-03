from ....pointer.pointer import Pointer
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


class RunFunctionOrConstructorAction(ImmediateActionWithoutReply):
    def __init__(self, path, args, kwargs, id_at_location, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.path = path
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location

    def execute_action(self, node: AbstractNode):
        method = node.lib_ast(self.path)

        resolved_args = list()
        for arg in self.args:
            if isinstance(arg, Pointer):
                r_arg = node.store.get_object(id=arg.id_at_location)
                resolved_args.append(r_arg)
            else:
                resolved_args.append(arg)

        resolved_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if isinstance(arg, Pointer):
                r_arg = node.store.get_object(id=arg.id_at_location)
                resolved_kwargs[arg_name] = r_arg
            else:
                resolved_kwargs[arg_name] = arg

        result = method(*resolved_args, **resolved_kwargs)

        node.store.store_object(id=self.id_at_location, obj=result)
