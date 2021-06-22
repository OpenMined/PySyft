# stdlib
import functools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
from syft.core.plan.plan import Plan

# syft relative
from ..... import lib
from ..... import serialize
from .....logger import critical
from .....logger import traceback_and_raise
from .....logger import warning
from .....proto.core.node.common.action.run_class_method_pb2 import (
    RunClassMethodAction as RunClassMethodAction_PB,
)
from .....util import inherit_tags
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


@bind_protobuf
class RunClassMethodAction(ImmediateActionWithoutReply):
    """
    When executing a RunClassMethodAction, a :class:`Node` will run a method defined
    by the action's path attribute on the object pointed at by _self and keep the returned
    value in its store.

    Attributes:
         path: the dotted path to the method to call
         _self: a pointer to the object which the method should be applied to.
         args: args to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
         kwargs: kwargs to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
    """

    def __init__(
        self,
        path: str,
        _self: Any,
        args: List[Any],
        kwargs: Dict[Any, Any],
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
        is_static: Optional[bool] = False,
    ):
        self.path = path
        self._self = _self
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location
        self.is_static = is_static
        # logging needs .path to exist before calling
        # this which is why i've put this super().__init__ down here
        super().__init__(address=address, msg_id=msg_id)

    @staticmethod
    def intersect_keys(
        left: Dict[VerifyKey, UID], right: Dict[VerifyKey, UID]
    ) -> Dict[VerifyKey, UID]:
        # get the intersection of the dict keys, the value is the request_id
        # if the request_id is different for some reason we still want to keep it,
        # so only intersect the keys and then copy those over from the main dict
        # into a new one
        intersection = set(left.keys()).intersection(right.keys())
        # left and right have the same keys
        return {k: left[k] for k in intersection}

    @property
    def pprint(self) -> str:
        return f"RunClassMethodAction({self.path})"

    def __repr__(self) -> str:
        method_name = self.path.split(".")[-1]
        self_name = self._self.class_name
        arg_names = ",".join([a.class_name for a in self.args])
        kwargs_names = ",".join([f"{k}={v.class_name}" for k, v in self.kwargs.items()])
        return f"RunClassMethodAction {self_name}.{method_name}({arg_names}, {kwargs_names})"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        method = node.lib_ast(self.path)

        mutating_internal = False
        if (
            self.path.startswith("torch.Tensor")
            and self.path.endswith("_")
            and not self.path.endswith("__call__")
        ):
            mutating_internal = True
        elif not self.path.startswith("torch.Tensor") and self.path.endswith(
            "__call__"
        ):
            mutating_internal = True

        resolved_self = None
        if not self.is_static:
            resolved_self = node.store.get_object(key=self._self.id_at_location)

            if resolved_self is None:
                critical(
                    f"execute_action on {self.path} failed due to missing object"
                    + f" at: {self._self.id_at_location}"
                )
                return
            result_read_permissions = resolved_self.read_permissions
        else:
            result_read_permissions = {}

        resolved_args = list()
        tag_args = []
        for arg in self.args:
            r_arg = node.store[arg.id_at_location]
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            resolved_args.append(r_arg.data)
            tag_args.append(r_arg)

        resolved_kwargs = {}
        tag_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            r_arg = node.store[arg.id_at_location]
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            resolved_kwargs[arg_name] = r_arg.data
            tag_kwargs[arg_name] = r_arg

        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(resolved_args, resolved_kwargs)

        if self.is_static:
            result = method(*upcasted_args, **upcasted_kwargs)
        else:
            if resolved_self is None:
                traceback_and_raise(
                    ValueError(f"Method {method} called, but self is None.")
                )

            method_name = self.path.split(".")[-1]

            if (
                isinstance(resolved_self.data, Plan)
                and method_name == "__call__"
                or (
                    hasattr(resolved_self.data, "forward")
                    and (
                        resolved_self.data.forward.__class__.__name__ == "Plan"
                        or getattr(resolved_self.data.forward, "__name__", None)
                        == "_compile_and_forward"
                    )
                    and method_name in ["__call__", "forward"]
                )
            ):

                if len(self.args) > 0:
                    traceback_and_raise(
                        ValueError(
                            "You passed args to Plan.__call__, while it only accepts kwargs"
                        )
                    )
                if method.__name__ == "_forward_unimplemented":
                    method = resolved_self.data.forward
                    result = method(node, verify_key, **self.kwargs)
                else:
                    result = method(resolved_self.data, node, verify_key, **self.kwargs)
            else:
                target_method = getattr(resolved_self.data, method_name, None)

                if id(target_method) != id(method):
                    warning(
                        f"Method {method_name} overwritten on object {resolved_self.data}"
                    )
                    method = target_method
                else:
                    method = functools.partial(method, resolved_self.data)

                result = method(*upcasted_args, **upcasted_kwargs)

        # TODO: add numpy support https://github.com/OpenMined/PySyft/issues/5164
        if "numpy." in str(type(result)):
            if "float" in type(result).__name__:
                result = float(result)
            if "int" in type(result).__name__:
                result = int(result)
            if "bool" in type(result).__name__:
                result = bool(result)

        if lib.python.primitive_factory.isprimitive(value=result):
            # Wrap in a SyPrimitive
            result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
                value=result, id=self.id_at_location
            )
        else:
            # TODO: overload all methods to incorporate this automatically
            if hasattr(result, "id"):
                try:
                    if hasattr(result, "_id"):
                        # set the underlying id
                        result._id = self.id_at_location
                    else:
                        result.id = self.id_at_location

                    if result.id != self.id_at_location:
                        raise AttributeError("IDs don't match")
                except AttributeError as e:
                    err = f"Unable to set id on result {type(result)}. {e}"
                    traceback_and_raise(Exception(err))

        if mutating_internal:
            if isinstance(resolved_self, StorableObject):
                resolved_self.read_permissions = result_read_permissions
        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
                read_permissions=result_read_permissions,
            )

        inherit_tags(
            attr_path_and_name=self.path,
            result=result,
            self_obj=resolved_self,
            args=tag_args,
            kwargs=tag_kwargs,
        )

        node.store[self.id_at_location] = result

    def _object2proto(self) -> RunClassMethodAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: RunClassMethodAction_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return RunClassMethodAction_PB(
            path=self.path,
            _self=serialize(self._self),
            args=list(map(lambda x: serialize(x), self.args)),
            kwargs={k: serialize(v) for k, v in self.kwargs.items()},
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            msg_id=serialize(self.id),
        )

    @staticmethod
    def _proto2object(proto: RunClassMethodAction_PB) -> "RunClassMethodAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of RunClassMethodAction
        :rtype: RunClassMethodAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return RunClassMethodAction(
            path=proto.path,
            _self=_deserialize(blob=proto._self),
            args=list(map(lambda x: _deserialize(blob=x), proto.args)),
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

        return RunClassMethodAction_PB

    def remap_input(self, current_input: Any, new_input: Any) -> None:
        """Redefines some of the arguments, and possibly the _self of the function"""
        if self._self.id_at_location == current_input.id_at_location:
            self._self = new_input

        for i, arg in enumerate(self.args):
            if arg.id_at_location == current_input.id_at_location:
                self.args[i] = new_input

        for k, v in self.kwargs.items():
            if v.id_at_location == current_input.id_at_location:
                self.kwargs[k] = new_input
