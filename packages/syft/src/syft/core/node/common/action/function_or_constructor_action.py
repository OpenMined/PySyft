# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey

# relative
from ..... import lib
from .....logger import traceback_and_raise
from .....util import inherit_tags
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....pointer.pointer import Pointer
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from ..util import check_send_to_blob_storage
from ..util import listify
from ..util import upload_result_to_s3
from .common import ImmediateActionWithoutReply
from .greenlets_switch import retrieve_object


@serializable(recursive_serde=True)
class RunFunctionOrConstructorAction(ImmediateActionWithoutReply):
    __attr_allowlist__ = ["path", "args", "kwargs", "id_at_location", "address", "id"]

    """
    When executing a RunFunctionOrConstructorAction, a :class:`Node` will run
    a function defined by the action's path attribute and keep the returned value
    in its store.

    Attributes:
         path: the dotted path to the function to call
         args: args to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
         kwargs: kwargs to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
    """

    def __init__(
        self,
        path: str,
        args: Union[Tuple[Any, ...], List[Any]],
        kwargs: Dict[Any, Any],
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
        is_static: Optional[bool] = False,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.path = path
        self.args = listify(args)  # args need to be editable for plans
        self.kwargs = kwargs
        self.id_at_location = id_at_location
        self.is_static = is_static

    @staticmethod
    def intersect_keys(
        left: Union[Dict[VerifyKey, Optional[UID]], None],
        right: Dict[VerifyKey, Optional[UID]],
    ) -> Dict[VerifyKey, Optional[UID]]:
        # TODO: duplicated in run_class_method_action.py
        # get the intersection of the dict keys, the value is the request_id
        # if the request_id is different for some reason we still want to keep it,
        # so only intersect the keys and then copy those over from the main dict
        # into a new one
        if left is None:
            return right
        intersection = left.keys() & right.keys()
        # left and right have the same keys
        return {k: left[k] for k in intersection}

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        method = node.lib_ast(self.path)
        result_read_permissions: Union[None, Dict[VerifyKey, Optional[UID]]] = None
        result_write_permissions: Union[None, Dict[VerifyKey, Optional[UID]]] = {
            verify_key: None
        }

        resolved_args = list()
        tag_args = []
        for arg in self.args:
            if not isinstance(arg, Pointer):
                traceback_and_raise(
                    ValueError(
                        f"args attribute of RunFunctionOrConstructorAction should only contain Pointers. "
                        f"Got {arg} of type {type(arg)}"
                    )
                )
            r_arg = retrieve_object(node, arg.id_at_location, self.path)
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )

            resolved_args.append(r_arg.data)
            tag_args.append(r_arg)

        resolved_kwargs = {}
        tag_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if not isinstance(arg, Pointer):
                traceback_and_raise(
                    ValueError(
                        f"kwargs attribute of RunFunctionOrConstructorAction should only contain Pointers. "
                        f"Got {arg} of type {type(arg)}"
                    )
                )
            r_arg = retrieve_object(node, arg.id_at_location, self.path)
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            resolved_kwargs[arg_name] = r_arg.data
            tag_kwargs[arg_name] = r_arg

        # upcast our args in case the method only accepts the original types
        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(resolved_args, resolved_kwargs)

        # execute the method with the newly upcasted args and kwargs
        result = method(*upcasted_args, **upcasted_kwargs)

        # to avoid circular imports
        if lib.python.primitive_factory.isprimitive(value=result):
            # Wrap in a SyPrimitive
            result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
                value=result, id=self.id_at_location
            )
        else:
            if hasattr(result, "id"):
                result._id = self.id_at_location

        # If we have no permission (None or {}) we add some default permissions based on a permission list
        if result_read_permissions is None:
            result_read_permissions = {}

        if result_write_permissions is None:
            result_write_permissions = {}

        # TODO: Upload object to seaweed store, instead of storing in redis
        # create a proxy object class and store it here.
        if check_send_to_blob_storage(
            obj=result,
            use_blob_storage=getattr(node.settings, "USE_BLOB_STORAGE", False),
        ):
            result = upload_result_to_s3(
                asset_name=self.id_at_location.no_dash,
                dataset_name="",
                domain_id=node.id,
                data=result,
                settings=node.settings,
            )

        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
                read_permissions=result_read_permissions,
                write_permissions=result_write_permissions,
            )

        inherit_tags(
            attr_path_and_name=self.path,
            result=result,
            self_obj=None,
            args=tag_args,
            kwargs=tag_kwargs,
        )

        node.store[self.id_at_location] = result

    def __repr__(self) -> str:
        method_name = self.path.split(".")[-1]
        arg_names = ",".join([a.__class__.__name__ for a in self.args])
        kwargs_names = ",".join(
            [f"{k}={v.__class__.__name__}" for k, v in self.kwargs.items()]
        )
        return f"FunctionOrConstructorAction {method_name}({arg_names}, {kwargs_names})"

    def remap_input(self, current_input: Any, new_input: Any) -> None:
        """Redefines some of the arguments of the function"""
        for i, arg in enumerate(self.args):
            if arg.id_at_location == current_input.id_at_location:
                self.args[i] = new_input

        for k, v in self.kwargs.items():
            if v.id_at_location == current_input.id_at_location:
                self.kwargs[k] = new_input
