# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.save_object_pb2 import (
    SaveObjectAction as SaveObjectAction_PB,
)
from ....common.group import VerifyAll
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import Serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


class SaveObjectAction(ImmediateActionWithoutReply, Serializable):
    @syft_decorator(typechecking=True)
    def __init__(
        self,
        id_at_location: UID,
        obj: object,
        address: Address,
        anyone_can_search_for_this: bool = False,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.id_at_location = id_at_location
        self.obj = obj
        self.anyone_can_search_for_this = anyone_can_search_for_this

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        # save the object to the store
        storable_obj = StorableObject(
            # ignoring this - the fundamental problem is that we can't force the classes
            # we want to use to subclass from something without creating wrappers for
            # everything which are mandatory for all operations. It's plausible that we
            # will have to do this - but for now we aren't so we need to simply assume
            # that we're adding ids to things. I don't like it though - wish there was a
            # better way. But we want to support other frameworks so - gotta do it.
            id=self.id_at_location,
            data=self.obj,
            tags=(
                # QUESTION: do we want None or an empty []
                self.obj.tags  # type: ignore
                if hasattr(self.obj, "tags")
                else None
            ),
            description=(
                self.obj.description  # type: ignore
                if hasattr(self.obj, "description")
                else ""
            ),
            search_permissions={VerifyAll(): None}
            if self.anyone_can_search_for_this
            else {},
            read_permissions={
                node.verify_key: node.id,
                verify_key: None,  # we dont have the passed in sender's UID
            },
        )

        node.store[self.id_at_location] = storable_obj

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SaveObjectAction_PB:

        id_at_location = self.id_at_location.serialize()
        obj_ob = self.obj.serialize()  # type: ignore
        addr = self.address.serialize()

        return SaveObjectAction_PB(
            id_at_location=id_at_location,
            obj=obj_ob,
            address=addr,
            anyone_can_search_for_this=self.anyone_can_search_for_this,
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: SaveObjectAction_PB) -> "SaveObjectAction":

        id_at_location = _deserialize(blob=proto.id_at_location)
        obj = _deserialize(blob=proto.obj)
        addr = _deserialize(blob=proto.address)
        anyone_can_search_for_this = proto.anyone_can_search_for_this

        return SaveObjectAction(
            id_at_location=id_at_location,
            obj=obj,
            address=addr,
            anyone_can_search_for_this=anyone_can_search_for_this,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SaveObjectAction_PB
