from .....proto.core.node.common.action.get_set_property_pb2 import GetOrSetPropertyAction as \
    GetOrSetPropertyAction_PB
# stdlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator


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
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
        set_arg: Optional[Any] = None
    ):
        super().__init__(address, msg_id)
        self.path = path
        self.id_at_location = id_at_location
        self.set_arg = set_arg
        self.msg_id = msg_id

    def intersect_keys(
        self, left: Union[Dict[VerifyKey, UID], None], right: Dict[VerifyKey, UID]
    ) -> Dict[VerifyKey, UID]:
        return RunClassMethodAction.intersect_keys(left, right)

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        property_ast_node = node.lib_ast.query(self.path)

        result = property_ast_node.get_property()

        result_read_permissions = {}
        # permissions rethinking

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
            set_arg=self.set_arg.serialize(),
            id_at_location=self.id_at_location.serialize(),
            address=self.address.serialize(),
            msg_id=self.id.serialize(),
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

        return GetOrSetPropertyAction_PB
