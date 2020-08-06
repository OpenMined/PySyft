from typing import Dict, Any, Tuple, Optional
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply

from syft.core.common.uid import UID
from syft.core.io.address import Address


class RunClassMethodAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        path: str,
        _self: Any,
        args: Tuple[Any],
        kwargs: Dict[Any, Any],
        id_at_location: int,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.path = path
        self._self = _self
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location

    def execute_action(self, node: AbstractNode) -> None:
        method = node.lib_ast(self.path)

        resolved_self = node.store.get_object(id=self._self.id_at_location)

        resolved_args = list()
        for arg in self.args:
            r_arg = node.store.get_object(id=arg.id_at_location)
            resolved_args.append(r_arg)

        resolved_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            r_arg = node.store.get_object(id=arg.id_at_location)
            resolved_kwargs[arg_name] = r_arg

        result = method(resolved_self, *resolved_args, **resolved_kwargs)

        node.store.store_object(id=self.id_at_location, obj=result)
