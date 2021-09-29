# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ...... import serialize
from ......proto.grid.messages.role_messages_pb2 import (
    CreateRoleMessage as CreateRoleMessage_PB,
)
from ......proto.grid.messages.role_messages_pb2 import (
    DeleteRoleMessage as DeleteRoleMessage_PB,
)
from ......proto.grid.messages.role_messages_pb2 import (
    GetRoleMessage as GetRoleMessage_PB,
)
from ......proto.grid.messages.role_messages_pb2 import (
    GetRoleResponse as GetRoleResponse_PB,
)
from ......proto.grid.messages.role_messages_pb2 import (
    GetRolesMessage as GetRolesMessage_PB,
)
from ......proto.grid.messages.role_messages_pb2 import (
    GetRolesResponse as GetRolesResponse_PB,
)
from ......proto.grid.messages.role_messages_pb2 import (
    UpdateRoleMessage as UpdateRoleMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable()
@final
class CreateRoleMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        name: str,
        reply_to: Address,
        can_make_data_requests: bool = False,
        can_triage_data_requests: bool = False,
        can_manage_privacy_budget: bool = False,
        can_create_users: bool = False,
        can_manage_users: bool = False,
        can_edit_roles: bool = False,
        can_manage_infrastructure: bool = False,
        can_upload_data: bool = False,
        can_upload_legal_document: bool = False,
        can_edit_domain_settings: bool = False,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.name = name

        self.can_make_data_requests = can_make_data_requests
        self.can_triage_data_requests = can_triage_data_requests
        self.can_manage_privacy_budget = can_manage_privacy_budget
        self.can_create_users = can_create_users
        self.can_manage_users = can_manage_users
        self.can_edit_roles = can_edit_roles
        self.can_manage_infrastructure = can_manage_infrastructure
        self.can_upload_data = can_upload_data
        self.can_upload_legal_document = can_upload_legal_document
        self.can_edit_domain_settings = can_edit_domain_settings

    def _object2proto(self) -> CreateRoleMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateRoleMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateRoleMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            name=self.name,
            can_make_data_requests=self.can_make_data_requests,
            can_triage_data_requests=self.can_triage_data_requests,
            can_manage_privacy_budget=self.can_manage_privacy_budget,
            can_create_users=self.can_create_users,
            can_manage_users=self.can_manage_users,
            can_edit_roles=self.can_edit_roles,
            can_manage_infrastructure=self.can_manage_infrastructure,
            can_upload_data=self.can_upload_data,
            can_upload_legal_document=self.can_upload_legal_document,
            can_edit_domain_settings=self.can_edit_domain_settings,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: CreateRoleMessage_PB,
    ) -> "CreateRoleMessage":
        """Creates a CreateRoleMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CreateRoleMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateRoleMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            name=proto.name,
            can_make_data_requests=proto.can_make_data_requests,
            can_triage_data_requests=proto.can_triage_data_requests,
            can_manage_privacy_budget=proto.can_manage_privacy_budget,
            can_create_users=proto.can_create_users,
            can_manage_users=proto.can_manage_users,
            can_edit_roles=proto.can_edit_roles,
            can_manage_infrastructure=proto.can_manage_infrastructure,
            can_upload_data=proto.can_upload_data,
            can_upload_legal_document=proto.can_upload_legal_document,
            can_edit_domain_settings=proto.can_edit_domain_settings,
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return CreateRoleMessage_PB


@serializable()
@final
class GetRoleMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        role_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.role_id = role_id

    def _object2proto(self) -> GetRoleMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetRoleMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRoleMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            role_id=self.role_id,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetRoleMessage_PB,
    ) -> "GetRoleMessage":
        """Creates a GetRoleMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetRoleMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetRoleMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            role_id=proto.role_id,
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRoleMessage_PB


@serializable()
@final
class GetRoleResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content

    def _object2proto(self) -> GetRoleResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRoleResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            content=serialize(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: GetRoleResponse_PB,
    ) -> "GetRoleResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetRoleResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=_deserialize(proto.content),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRoleResponse_PB


@serializable()
@final
class GetRolesMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetRolesMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetRolesMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetRolesMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetRolesMessage_PB,
    ) -> "GetRolesMessage":
        """Creates a GetRolesMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetRolesMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetRolesMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRolesMessage_PB


@serializable()
@final
class GetRolesResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content

    def _object2proto(self) -> GetRolesResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        msg = GetRolesResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )
        _ = [msg.content.append(serialize(content)) for content in self.content]
        return msg

    @staticmethod
    def _proto2object(
        proto: GetRolesResponse_PB,
    ) -> "GetRolesResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return GetRolesResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=[_deserialize(content) for content in proto.content],
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetRolesResponse_PB


@serializable()
@final
class UpdateRoleMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        role_id: int,
        name: str,
        reply_to: Address,
        can_make_data_requests: bool = False,
        can_triage_data_requests: bool = False,
        can_manage_privacy_budget: bool = False,
        can_create_users: bool = False,
        can_manage_users: bool = False,
        can_edit_roles: bool = False,
        can_manage_infrastructure: bool = False,
        can_upload_data: bool = False,
        can_upload_legal_document: bool = False,
        can_edit_domain_settings: bool = False,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.name = name
        self.can_make_data_requests = can_make_data_requests
        self.can_triage_data_requests = can_triage_data_requests
        self.can_manage_privacy_budget = can_manage_privacy_budget
        self.can_create_users = can_create_users
        self.can_manage_users = can_manage_users
        self.can_edit_roles = can_edit_roles
        self.can_manage_infrastructure = can_manage_infrastructure
        self.can_upload_data = can_upload_data
        self.can_upload_legal_document = can_upload_legal_document
        self.can_edit_domain_settings = can_edit_domain_settings
        self.role_id = role_id

    def _object2proto(self) -> UpdateRoleMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: UpdateRoleMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateRoleMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            name=self.name,
            can_make_data_requests=self.can_make_data_requests,
            can_triage_data_requests=self.can_triage_data_requests,
            can_manage_privacy_budget=self.can_manage_privacy_budget,
            can_create_users=self.can_create_users,
            can_manage_users=self.can_manage_users,
            can_edit_roles=self.can_edit_roles,
            can_manage_infrastructure=self.can_manage_infrastructure,
            can_upload_data=self.can_upload_data,
            can_upload_legal_document=self.can_upload_legal_document,
            can_edit_domain_settings=self.can_edit_domain_settings,
            role_id=self.role_id,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateRoleMessage_PB,
    ) -> "UpdateRoleMessage":
        """Creates a UpdateRoleMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: UpdateRoleMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateRoleMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            name=proto.name,
            can_make_data_requests=proto.can_make_data_requests,
            can_triage_data_requests=proto.can_triage_data_requests,
            can_manage_privacy_budget=proto.can_manage_privacy_budget,
            can_create_users=proto.can_create_users,
            can_manage_users=proto.can_manage_users,
            can_edit_roles=proto.can_edit_roles,
            can_manage_infrastructure=proto.can_manage_infrastructure,
            can_upload_data=proto.can_upload_data,
            can_upload_legal_document=proto.can_upload_legal_document,
            can_edit_domain_settings=proto.can_edit_domain_settings,
            role_id=proto.role_id,
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return UpdateRoleMessage_PB


@serializable()
@final
class DeleteRoleMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        role_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.role_id = role_id

    def _object2proto(self) -> DeleteRoleMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: DeleteRoleMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return DeleteRoleMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            role_id=self.role_id,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: DeleteRoleMessage_PB,
    ) -> "DeleteRoleMessage":
        """Creates a DeleteRoleMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: DeleteRoleMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return DeleteRoleMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            role_id=proto.role_id,
            reply_to=_deserialize(blob=proto.reply_to),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return DeleteRoleMessage_PB
