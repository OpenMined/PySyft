# external class imports
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from nacl.signing import VerifyKey
from .common import ImmediateActionWithoutReply
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft imports
# from .....lib.python.primitive import isprimitive, PyPrimitive

from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from ....store.storeable_object import StorableObject
from ....common.serde.deserialize import _deserialize
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.run_class_method_pb2 import (
    RunClassMethodAction as RunClassMethodAction_PB,
)


class RunClassMethodAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        path: str,
        _self: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[Any, Any],
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        self.path = path
        self._self = _self
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location

        # logging (sy.VERBOSE) needs .path to exist before calling
        # this which is why i've put this super().__init__ down here
        super().__init__(address=address, msg_id=msg_id)

    def intersect_keys(
        self, left: Dict[VerifyKey, UID], right: Dict[VerifyKey, UID]
    ) -> Dict[VerifyKey, UID]:
        # get the intersection of the dict keys, the value is the request_id
        # if the request_id is different for some reason we still want to keep it,
        # so only intersect the keys and then copy those over from the main dict
        # into a new one
        intersection = set(left.keys()).intersection(right.keys())
        intersection_dict = {}
        for k in intersection:
            intersection_dict[k] = left[k]  # left and right have the same keys

        return intersection_dict

    @property
    def pprint(self) -> str:
        return f"RunClassMethodAction({self.path})"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        method = node.lib_ast(self.path)

        resolved_self = node.store[self._self.id_at_location]

        result_read_permissions = resolved_self.read_permissions

        resolved_args = list()
        for arg in self.args:
            r_arg = node.store[arg.id_at_location]

            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )

            resolved_args.append(r_arg.data)

        resolved_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            r_arg = node.store[arg.id_at_location]
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            resolved_kwargs[arg_name] = r_arg.data

        result = method(resolved_self.data, *resolved_args, **resolved_kwargs)

        # if isprimitive(value=result):
        #     # Wrap in a PyPrimitive
        #     result = PyPrimitive(data=result, id=self.id_at_location)
        # else:
        # TODO: overload all methods to incorporate this automatically
        if hasattr(result, "id"):
            result.id = self.id_at_location

        # QUESTION: There seems to be return value tensors that have no id
        # and get auto wrapped? So is this code not correct?
        # else:
        #     # TODO: Add tests, this could happen if the isprimitive fails due to an
        #     # unsupported type
        #     raise Exception(f"Result has no ID. {result}")

        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
                read_permissions=result_read_permissions,
            )
        node.store.store(obj=result)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RunClassMethodAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: RunClassMethodAction_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return RunClassMethodAction_PB(
            path=self.path,
            _self=self._self.serialize(),
            args=list(map(lambda x: x.serialize(), self.args)),
            kwargs={k: v.serialize() for k, v in self.kwargs.items()},
            id_at_location=self.id_at_location.serialize(),
            address=self.address.serialize(),
            msg_id=self.id.serialize(),
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
            args=tuple(map(lambda x: _deserialize(blob=x), proto.args)),
            kwargs={k: _deserialize(blob=v) for k, v in proto.kwargs.items()},
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """ Return the type of protobuf object which stores a class of this type

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
