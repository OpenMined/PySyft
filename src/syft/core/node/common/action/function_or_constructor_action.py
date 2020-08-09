from typing import Tuple, Any, Dict, Optional
from ....pointer.pointer import Pointer
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply

from syft.core.common.uid import UID
from syft.core.io.address import Address

from ....store.storeable_object import StorableObject

class RunFunctionOrConstructorAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        path: str,
        args: Tuple[Any],
        kwargs: Dict[Any, Any],
        id_at_location: int,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.path = path
        self.args = args
        self.kwargs = kwargs

        # TODO: eliminate this explicit parameter and just set the object
        #  id on the object directly.
        self.id_at_location = id_at_location

    def execute_action(self, node: AbstractNode) -> None:
        method = node.lib_ast(self.path)

        resolved_args = list()
        for arg in self.args:
            if isinstance(arg, Pointer):
                # QUESTION: Where is get_object definied?
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

        if not isinstance(result, StorableObject):
            result = StorableObject(id=self.id_at_location,
                                    data=result)

        # QUESTION: Where is store_object defined
        node.store.store(obj=result)
