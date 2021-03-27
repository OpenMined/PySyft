# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ..... import lib
from ..... import serialize
from .....logger import traceback_and_raise
from .....proto.core.node.common.action.run_function_or_constructor_pb2 import (
    RunFunctionOrConstructorAction as RunFunctionOrConstructorAction_PB,
)
from .....util import inherit_tags
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ....pointer.pointer import Pointer
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from ..util import listify
from .common import ImmediateActionWithoutReply


@bind_protobuf
class RunFunctionOrConstructorAction(ImmediateActionWithoutReply):
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
        left: Union[Dict[VerifyKey, UID], None], right: Dict[VerifyKey, UID]
    ) -> Dict[VerifyKey, UID]:
        # FIXME duplicated in run_class_method_action.py
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
        result_read_permissions: Union[None, Dict[VerifyKey, UID]] = None

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

            r_arg = node.store[arg.id_at_location]
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

            r_arg = node.store[arg.id_at_location]
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

        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
                read_permissions=result_read_permissions,
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
        arg_names = ",".join([a.class_name for a in self.args])
        kwargs_names = ",".join([f"{k}={v.class_name}" for k, v in self.kwargs.items()])
        return f"RunClassMethodAction {method_name}({arg_names}, {kwargs_names})"

    def _object2proto(self) -> RunFunctionOrConstructorAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: RunFunctionOrConstructorAction_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return RunFunctionOrConstructorAction_PB(
            path=self.path,
            args=[serialize(x) for x in self.args],
            kwargs={k: serialize(v) for k, v in self.kwargs.items()},
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            msg_id=serialize(self.id),
        )

    @staticmethod
    def _proto2object(
        proto: RunFunctionOrConstructorAction_PB,
    ) -> "RunFunctionOrConstructorAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of RunFunctionOrConstructorAction
        :rtype: RunFunctionOrConstructorAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return RunFunctionOrConstructorAction(
            path=proto.path,
            args=tuple(_deserialize(blob=x) for x in proto.args),
            kwargs={k: _deserialize(blob=v) for k, v in proto.kwargs.items()},
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return RunFunctionOrConstructorAction_PB

    def remap_input(self, current_input: Any, new_input: Any) -> None:
        """Redefines some of the arguments of the function"""
        for i, arg in enumerate(self.args):
            if arg.id_at_location == current_input.id_at_location:
                self.args[i] = new_input

        for k, v in self.kwargs.items():
            if v.id_at_location == current_input.id_at_location:
                self.kwargs[k] = new_input
