# stdlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ..... import lib
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.get_set_property_pb2 import (
    GetOrSetPropertyAction as GetOrSetPropertyAction_PB,
)
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .run_class_method_action import RunClassMethodAction


class GetOrSetPropertyAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        path: str,
        _self: Any,
        id_at_location: UID,
        address: Address,
        set_arg: Optional[Any] = None,
    ):
        super().__init__(address)
        self.path = path
        self.id_at_location = id_at_location
        self.set_arg = set_arg
        self._self = _self

    def intersect_keys(
        self, left: Union[Dict[VerifyKey, UID], None], right: Dict[VerifyKey, UID]
    ) -> Dict[VerifyKey, UID]:
        return RunClassMethodAction.intersect_keys(left, right)

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        method = node.lib_ast(self.path)

        # TODO: properly mark this in the ast, don't leave the action to decide this
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

        resolved_self = node.store.get_object(key=self._self.id_at_location)
        if resolved_self is None:
            logger.critical(
                f"execute_action on {self.path} failed due to missing object"
                + f" at: {self._self.id_at_location}"
            )
            return

        result_read_permissions = resolved_self.read_permissions

        if type(method).__name__ in ["getset_descriptor", "_tuplegetter"]:
            # we have a detached class property so we need the __get__ descriptor
            upcast_attr = getattr(resolved_self.data, "upcast", None)
            data = resolved_self.data
            if upcast_attr is not None:
                data = upcast_attr()

            result = method.__get__(data)
        else:
            # we have a callable
            # upcast our args in case the method only accepts the original types
            (
                upcasted_args,
                upcasted_kwargs,
            ) = lib.python.util.upcast_args_and_kwargs(resolved_args, resolved_kwargs)
            result = method(resolved_self.data, *upcasted_args, **upcasted_kwargs)

        # to avoid circular imports

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

                    assert result.id == self.id_at_location
                except AttributeError as e:
                    err = f"Unable to set id on result {type(result)}. {e}"
                    raise Exception(err)

        # QUESTION: There seems to be return value tensors that have no id
        # and get auto wrapped? So is this code not correct?
        # else:
        #     # TODO: Add tests, this could happen if the isprimitive fails due to an
        #     # unsupported type
        #     raise Exception(f"Result has no ID. {result}")

        if mutating_internal:
            resolved_self.read_permissions = result_read_permissions
        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
                read_permissions=result_read_permissions,
            )

        node.store[self.id_at_location] = result

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetOrSetPropertyAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetOrSetPropertyAction_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetOrSetPropertyAction_PB(
            path=self.path,
            id_at_location=self.id_at_location.serialize(),
            address=self.address.serialize(),
            _self=self._self.serialize(),
        )

    @staticmethod
    def _proto2object(
        proto: GetOrSetPropertyAction_PB,
    ) -> "GetOrSetPropertyAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetOrSetPropertyAction
        :rtype: GetOrSetPropertyAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetOrSetPropertyAction(
            path=proto.path,
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            _self=_deserialize(blob=proto._self),
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

        return GetOrSetPropertyAction_PB
