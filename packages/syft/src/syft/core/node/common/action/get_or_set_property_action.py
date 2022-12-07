# stdlib
from enum import Enum
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey

# relative
from ..... import lib
from .....util import inherit_tags
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .greenlets_switch import retrieve_object
from .run_class_method_action import RunClassMethodAction


class PropertyActions(Enum):
    SET = 1
    GET = 2
    DEL = 3


@serializable(recursive_serde=True)
class GetOrSetPropertyAction(ImmediateActionWithoutReply):
    __attr_allowlist__ = [
        "path",
        "id_at_location",
        "args",
        "kwargs",
        "address",
        "_self",
        "id",
        "action",
        "map_to_dyn",
    ]

    __serde_overrides__: Dict[str, Sequence[Callable]] = {
        "action": (lambda x: int(x.value), lambda x: PropertyActions(int(x)))
    }

    def __init__(
        self,
        path: str,
        _self: Any,
        id_at_location: UID,
        address: Address,
        args: Union[Tuple[Any, ...], List[Any]],
        kwargs: Dict[Any, Any],
        action: PropertyActions,
        map_to_dyn: bool,
        set_arg: Optional[Any] = None,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address, msg_id=msg_id)
        self.path = path
        self.id_at_location = id_at_location
        self.set_arg = set_arg
        self._self = _self
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.map_to_dyn = map_to_dyn

    def intersect_keys(
        self,
        left: Dict[VerifyKey, Optional[UID]],
        right: Dict[VerifyKey, Optional[UID]],
    ) -> Dict[VerifyKey, Optional[UID]]:
        return RunClassMethodAction.intersect_keys(left, right)

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        ast_node = node.lib_ast.query(self.path)
        method = ast_node.object_ref

        # storable object raw from object store
        resolved_self = retrieve_object(node, self._self.id_at_location, self.path)
        result_read_permissions = resolved_self.read_permissions

        resolved_args = []
        tag_args = []
        for arg in self.args:
            r_arg = retrieve_object(node, arg.id_at_location, self.path)
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            tag_args.append(r_arg)
            resolved_args.append(r_arg.data)

        resolved_kwargs = {}
        tag_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            r_arg = retrieve_object(node, arg.id_at_location, self.path)
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            tag_kwargs[arg_name] = r_arg
            resolved_kwargs[arg_name] = r_arg.data

        if not (inspect.isdatadescriptor(method) or self.map_to_dyn):
            raise ValueError(f"{method} not an actual property!")

        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(resolved_args, resolved_kwargs)

        data = resolved_self.data

        if self.map_to_dyn:
            if self.action == PropertyActions.SET:
                setattr(data, ast_node.name, *upcasted_args)
                result = None
                # since we may have changed resolve_self.data we need to check it back in
                node.store[self._self.id_at_location] = resolved_self
            elif self.action == PropertyActions.GET:
                result = getattr(data, ast_node.name)
            elif self.action == PropertyActions.DEL:
                raise ValueError(f"{self.action} not a valid action!")
        else:
            if self.action == PropertyActions.SET:
                result = method.__set__(data, *upcasted_args, **upcasted_kwargs)

                # since we may have changed resolve_self.data we need to check it back in
                node.store[self._self.id_at_location] = resolved_self

            elif self.action == PropertyActions.GET:
                result = method.__get__(data, *upcasted_args, **upcasted_kwargs)
            elif self.action == PropertyActions.DEL:
                result = method.__del__(data, *upcasted_args, **upcasted_kwargs)

                # since we may have changed resolve_self.data we need to check it back in
                node.store[self._self.id_at_location] = resolved_self
            else:
                raise ValueError(f"{self.action} not a valid action!")

        if lib.python.primitive_factory.isprimitive(value=result):
            result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
                value=result, id=self.id_at_location
            )
        else:
            if hasattr(result, "id"):
                # just to make mypy to shut up, impossible case
                if result is None:
                    raise Exception("Please convert None so _SyNone.")

                try:
                    if hasattr(result, "_id"):
                        # set the underlying id
                        result._id = self.id_at_location
                    else:
                        result.id = self.id_at_location

                    if result.id != self.id_at_location:
                        raise AttributeError("IDs don't match")
                except AttributeError:
                    raise Exception("MAKE VALID SCHEMA")

        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
                read_permissions=result_read_permissions,
            )

        # When GET, result is a new object, we give new tags to it
        if self.action == PropertyActions.GET:
            inherit_tags(
                attr_path_and_name=self.path,
                result=result,
                self_obj=resolved_self,
                args=tag_args,
                kwargs=tag_kwargs,
            )

        node.store[self.id_at_location] = result

    def __repr__(self) -> str:
        attr_name = self.path.split(".")[-1]
        self_name = str(self._self.__class__.__name__)

        if self.action == PropertyActions.SET:
            val = self.args[0]
            return f"GetOrSetPropertyAction {self_name}.{attr_name} = {val}"
        elif self.action == PropertyActions.GET:
            return f"GetOrSetPropertyAction GET {self_name}.{attr_name}"
        else:
            return f"GetOrSetPropertyAction DEL {self_name}.{attr_name}"
