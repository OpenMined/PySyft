# stdlib
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply

# from syft.grid.messages.setup_messages import CreateInitialSetUpMessage
# from syft.grid.messages.setup_messages import GetSetUpMessage
# from syft.grid.messages.setup_messages import GetSetUpResponse
from syft.grid.messages.success_resp_message import SuccessResponseMessage

# relative
from .....logger import traceback_and_raise
from ..exceptions import AuthorizationError
from ..exceptions import InvalidParameterValueError
from ..exceptions import MissingRequestKeyError
from ..exceptions import OwnerAlreadyExistsError
from ..tables.setup import SetupConfig
from ..tables.utils import model_to_json


# stdlib
import json
from typing import Dict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft absolute
from syft import serialize
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.serde.serializable import bind_protobuf
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.proto.grid.messages.setup_messages_pb2 import (
    CreateInitialSetUpMessage as CreateInitialSetUpMessage_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    GetSetUpMessage as GetSetUpMessage_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    GetSetUpResponse as GetSetUpResponse_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    UpdateSetupMessage as UpdateSetupMessage_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    UpdateSetupResponse as UpdateSetupResponse_PB,
)


@bind_protobuf
@final
class GetSetUpMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    def _object2proto(self) -> GetSetUpMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetSetUpMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetSetUpMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            content=json.dumps(self.content),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetSetUpMessage_PB,
    ) -> "GetSetUpMessage":
        """Creates a GetSetUpMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetSetUpMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetSetUpMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=json.loads(proto.content),
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

        return GetSetUpMessage_PB


@bind_protobuf
@final
class GetSetUpResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content

    def _object2proto(self) -> GetSetUpResponse_PB:
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
        return GetSetUpResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: GetSetUpResponse_PB,
    ) -> "GetSetUpResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetSetUpResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            content=json.loads(proto.content),
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

        return GetSetUpResponse_PB


@bind_protobuf
@final
class CreateInitialSetUpMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        email: str,
        password: str,
        domain_name: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.email = email
        self.password = password
        self.domain_name = domain_name

    def _object2proto(self) -> CreateInitialSetUpMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateInitialSetUpMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateInitialSetUpMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            email=self.email,
            password=self.password,
            domain_name=self.domain_name,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: CreateInitialSetUpMessage_PB,
    ) -> "CreateInitialSetUpMessage":
        """Creates a CreateInitialSetUpMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CreateInitialSetUpMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateInitialSetUpMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            email=proto.email,
            password=proto.password,
            domain_name=proto.domain_name,
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

        return CreateInitialSetUpMessage_PB


@bind_protobuf
@final
class UpdateSetupMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    def _object2proto(self) -> UpdateSetupMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: UpdateSetupMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateSetupMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            content=json.dumps(self.content),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateSetupMessage_PB,
    ) -> "UpdateSetupMessage":
        """Creates a UpdateSetupMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: UpdateSetupMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateSetupMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=json.loads(proto.content),
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

        return UpdateSetupMessage_PB


@bind_protobuf
@final
class UpdateSetupResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content

    def _object2proto(self) -> UpdateSetupResponse_PB:
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
        return UpdateSetupResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateSetupResponse_PB,
    ) -> "UpdateSetupResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateSetupResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            content=json.loads(proto.content),
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

        return UpdateSetupResponse_PB


def create_initial_setup(
    msg: CreateInitialSetUpMessage, node: AbstractNode, verify_key: VerifyKey
) -> SuccessResponseMessage:
    # 1 - Should not run if Node has an owner
    if len(node.users):
        raise OwnerAlreadyExistsError

    # 2 - Check if email/password/node_name fields are empty
    _mandatory_request_fields = msg.email and msg.password and msg.domain_name
    if not _mandatory_request_fields:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (email/password/domain_name)!"
        )

    # 3 - Change Node Name
    node.name = msg.domain_name

    # 4 - Create Admin User
    _node_private_key = node.signing_key.encode(encoder=HexEncoder).decode("utf-8")
    _verify_key = node.signing_key.verify_key.encode(encoder=HexEncoder).decode("utf-8")
    _admin_role = node.roles.owner_role
    _ = node.users.signup(
        email=msg.email,
        password=msg.password,
        role=_admin_role.id,
        private_key=_node_private_key,
        verify_key=_verify_key,
    )

    # 5 - Save Node SetUp Configs
    node.setup.register(domain_name=msg.domain_name)

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Running initial setup!",
    )


def get_setup(
    msg: GetSetUpMessage, node: AbstractNode, verify_key: VerifyKey
) -> GetSetUpResponse:

    _current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not _current_user_id:
        try:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id
        except Exception as e:
            traceback_and_raise(e)

    if users.role(user_id=_current_user_id).name != "Owner":
        raise AuthorizationError("You're not allowed to get setup configs!")
    else:
        _setup = model_to_json(node.setup.first(domain_name=node.name))

    return GetSetUpResponse(
        address=msg.reply_to,
        status_code=200,
        content=_setup,
    )


class SetUpService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateInitialSetUpMessage: create_initial_setup,
        GetSetUpMessage: get_setup,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[SuccessResponseMessage, GetSetUpResponse,]:
        return SetUpService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ]
