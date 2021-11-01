# # stdlib
# import json
# from typing import Dict
# from typing import Optional
#
# # third party
# from google.protobuf.reflection import GeneratedProtocolMessageType
# from typing_extensions import final
#
# # syft absolute
# from syft import serialize
# from syft.core.common.message import ImmediateSyftMessageWithReply
# from syft.core.common.message import ImmediateSyftMessageWithoutReply
# from syft.core.common.serde.deserialize import _deserialize
# from syft.core.common.serde.serializable import serializable
# from syft.core.common.uid import UID
# from syft.core.io.address import Address
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     CreateWorkerMessage as CreateWorkerMessage_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     CreateWorkerResponse as CreateWorkerResponse_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     DeleteWorkerMessage as DeleteWorkerMessage_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     DeleteWorkerResponse as DeleteWorkerResponse_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     GetWorkerInstanceTypesMessage as GetWorkerInstanceTypesMessage_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     GetWorkerInstanceTypesResponse as GetWorkerInstanceTypesResponse_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     GetWorkerMessage as GetWorkerMessage_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     GetWorkerResponse as GetWorkerResponse_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     GetWorkersMessage as GetWorkersMessage_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     GetWorkersResponse as GetWorkersResponse_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     UpdateWorkerMessage as UpdateWorkerMessage_PB,
# )
# from syft.proto.grid.messages.infra_messages_pb2 import (
#     UpdateWorkerResponse as UpdateWorkerResponse_PB,
# )
#
#
# @serializable()
# @final
# class CreateWorkerMessage(ImmediateSyftMessageWithReply):
#     def __init__(
#         self,
#         address: Address,
#         content: Dict,
#         reply_to: Address,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
#         self.content = content
#
#     def _object2proto(self) -> CreateWorkerMessage_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: CreateWorkerMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return CreateWorkerMessage_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             content=json.dumps(self.content),
#             reply_to=serialize(self.reply_to),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: CreateWorkerMessage_PB,
#     ) -> "CreateWorkerMessage":
#         """Creates a CreateWorkerMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: CreateWorkerMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return CreateWorkerMessage(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             content=json.loads(proto.content),
#             reply_to=_deserialize(blob=proto.reply_to),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return CreateWorkerMessage_PB
#
#
# @serializable()
# @final
# class CreateWorkerResponse(ImmediateSyftMessageWithoutReply):
#     def __init__(
#         self,
#         address: Address,
#         status_code: int,
#         content: Dict,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id)
#         self.status_code = status_code
#         self.content = content
#
#     def _object2proto(self) -> CreateWorkerResponse_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: SignalingOfferMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return CreateWorkerResponse_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             status_code=self.status_code,
#             content=json.dumps(self.content),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: CreateWorkerResponse_PB,
#     ) -> "CreateWorkerResponse":
#         """Creates a SignalingOfferMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: SignalingOfferMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return CreateWorkerResponse(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             status_code=proto.status_code,
#             content=json.loads(proto.content),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return CreateWorkerResponse_PB
#
#
# @serializable()
# @final
# class GetWorkerInstanceTypesMessage(ImmediateSyftMessageWithReply):
#     def __init__(
#         self,
#         address: Address,
#         content: Dict,
#         reply_to: Address,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
#         self.content = content
#
#     def _object2proto(self) -> GetWorkerInstanceTypesMessage_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: GetWorkerInstanceTypesMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return GetWorkerInstanceTypesMessage_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             content=json.dumps(self.content),
#             reply_to=serialize(self.reply_to),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: GetWorkerInstanceTypesMessage_PB,
#     ) -> "GetWorkerInstanceTypesMessage":
#         """Creates a GetWorkerInstanceTypesMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: GetWorkerInstanceTypesMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return GetWorkerInstanceTypesMessage(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             content=json.loads(proto.content),
#             reply_to=_deserialize(blob=proto.reply_to),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return GetWorkerInstanceTypesMessage_PB
#
#
# @serializable()
# @final
# class GetWorkerInstanceTypesResponse(ImmediateSyftMessageWithoutReply):
#     def __init__(
#         self,
#         address: Address,
#         status_code: int,
#         content: Dict,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id)
#         self.status_code = status_code
#         self.content = content
#
#     def _object2proto(self) -> GetWorkerInstanceTypesResponse_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: SignalingOfferMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return GetWorkerInstanceTypesResponse_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             status_code=self.status_code,
#             content=json.dumps(self.content),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: GetWorkerInstanceTypesResponse_PB,
#     ) -> "GetWorkerInstanceTypesResponse":
#         """Creates a SignalingOfferMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: SignalingOfferMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return GetWorkerInstanceTypesResponse(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             status_code=proto.status_code,
#             content=json.loads(proto.content),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return GetWorkerInstanceTypesResponse_PB
#
#
# @serializable()
# @final
# class GetWorkerMessage(ImmediateSyftMessageWithReply):
#     def __init__(
#         self,
#         address: Address,
#         content: Dict,
#         reply_to: Address,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
#         self.content = content
#
#     def _object2proto(self) -> GetWorkerMessage_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: GetWorkerMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return GetWorkerMessage_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             content=json.dumps(self.content),
#             reply_to=serialize(self.reply_to),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: GetWorkerMessage_PB,
#     ) -> "GetWorkerMessage":
#         """Creates a GetWorkerMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: GetWorkerMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return GetWorkerMessage(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             content=json.loads(proto.content),
#             reply_to=_deserialize(blob=proto.reply_to),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return GetWorkerMessage_PB
#
#
# @serializable()
# @final
# class GetWorkerResponse(ImmediateSyftMessageWithoutReply):
#     def __init__(
#         self,
#         address: Address,
#         status_code: int,
#         content: Dict,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id)
#         self.status_code = status_code
#         self.content = content
#
#     def _object2proto(self) -> GetWorkerResponse_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: SignalingOfferMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return GetWorkerResponse_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             status_code=self.status_code,
#             content=json.dumps(self.content),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: GetWorkerResponse_PB,
#     ) -> "GetWorkerResponse":
#         """Creates a SignalingOfferMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: SignalingOfferMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return GetWorkerResponse(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             status_code=proto.status_code,
#             content=json.loads(proto.content),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return GetWorkerResponse_PB
#
#
# @serializable()
# @final
# class GetWorkersMessage(ImmediateSyftMessageWithReply):
#     def __init__(
#         self,
#         address: Address,
#         content: Dict,
#         reply_to: Address,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
#         self.content = content
#
#     def _object2proto(self) -> GetWorkersMessage_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: GetWorkersMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return GetWorkersMessage_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             content=json.dumps(self.content),
#             reply_to=serialize(self.reply_to),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: GetWorkersMessage_PB,
#     ) -> "GetWorkersMessage":
#         """Creates a GetWorkersMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: GetWorkersMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return GetWorkersMessage(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             content=json.loads(proto.content),
#             reply_to=_deserialize(blob=proto.reply_to),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return GetWorkersMessage_PB
#
#
# @serializable()
# @final
# class GetWorkersResponse(ImmediateSyftMessageWithoutReply):
#     def __init__(
#         self,
#         address: Address,
#         status_code: int,
#         content: Dict,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id)
#         self.status_code = status_code
#         self.content = content
#
#     def _object2proto(self) -> GetWorkersResponse_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: SignalingOfferMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return GetWorkersResponse_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             status_code=self.status_code,
#             content=json.dumps(self.content),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: GetWorkersResponse_PB,
#     ) -> "GetWorkersResponse":
#         """Creates a SignalingOfferMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: SignalingOfferMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return GetWorkersResponse(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             status_code=proto.status_code,
#             content=json.loads(proto.content),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return GetWorkersResponse_PB
#
#
# @serializable()
# @final
# class UpdateWorkerMessage(ImmediateSyftMessageWithReply):
#     def __init__(
#         self,
#         address: Address,
#         content: Dict,
#         reply_to: Address,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
#         self.content = content
#
#     def _object2proto(self) -> UpdateWorkerMessage_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: UpdateWorkerMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return UpdateWorkerMessage_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             content=json.dumps(self.content),
#             reply_to=serialize(self.reply_to),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: UpdateWorkerMessage_PB,
#     ) -> "UpdateWorkerMessage":
#         """Creates a UpdateWorkerMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: UpdateWorkerMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return UpdateWorkerMessage(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             content=json.loads(proto.content),
#             reply_to=_deserialize(blob=proto.reply_to),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return UpdateWorkerMessage_PB
#
#
# @serializable()
# @final
# class UpdateWorkerResponse(ImmediateSyftMessageWithoutReply):
#     def __init__(
#         self,
#         address: Address,
#         status_code: int,
#         content: Dict,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id)
#         self.status_code = status_code
#         self.content = content
#
#     def _object2proto(self) -> UpdateWorkerResponse_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: SignalingOfferMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return UpdateWorkerResponse_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             status_code=self.status_code,
#             content=json.dumps(self.content),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: UpdateWorkerResponse_PB,
#     ) -> "UpdateWorkerResponse":
#         """Creates a SignalingOfferMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: SignalingOfferMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return UpdateWorkerResponse(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             status_code=proto.status_code,
#             content=json.loads(proto.content),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return UpdateWorkerResponse_PB
#
#
# @serializable()
# @final
# class DeleteWorkerMessage(ImmediateSyftMessageWithReply):
#     def __init__(
#         self,
#         address: Address,
#         content: Dict,
#         reply_to: Address,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
#         self.content = content
#
#     def _object2proto(self) -> DeleteWorkerMessage_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: DeleteWorkerMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return DeleteWorkerMessage_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             content=json.dumps(self.content),
#             reply_to=serialize(self.reply_to),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: DeleteWorkerMessage_PB,
#     ) -> "DeleteWorkerMessage":
#         """Creates a DeleteWorkerMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: DeleteWorkerMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return DeleteWorkerMessage(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             content=json.loads(proto.content),
#             reply_to=_deserialize(blob=proto.reply_to),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return DeleteWorkerMessage_PB
#
#
# @serializable()
# @final
# class DeleteWorkerResponse(ImmediateSyftMessageWithoutReply):
#     def __init__(
#         self,
#         address: Address,
#         status_code: int,
#         content: Dict,
#         msg_id: Optional[UID] = None,
#     ):
#         super().__init__(address=address, msg_id=msg_id)
#         self.status_code = status_code
#         self.content = content
#
#     def _object2proto(self) -> DeleteWorkerResponse_PB:
#         """Returns a protobuf serialization of self.
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#         :return: returns a protobuf object
#         :rtype: SignalingOfferMessage_PB
#         .. note::
#             This method is purely an internal method. Please use serialize(object) or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#         return DeleteWorkerResponse_PB(
#             msg_id=serialize(self.id),
#             address=serialize(self.address),
#             status_code=self.status_code,
#             content=json.dumps(self.content),
#         )
#
#     @staticmethod
#     def _proto2object(
#         proto: DeleteWorkerResponse_PB,
#     ) -> "DeleteWorkerResponse":
#         """Creates a SignalingOfferMessage from a protobuf
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#         :return: returns an instance of SignalingOfferMessage
#         :rtype: SignalingOfferMessage
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         return DeleteWorkerResponse(
#             msg_id=_deserialize(blob=proto.msg_id),
#             address=_deserialize(blob=proto.address),
#             status_code=proto.status_code,
#             content=json.loads(proto.content),
#         )
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """Return the type of protobuf object which stores a class of this type
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for
#         details.
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#         """
#
#         return DeleteWorkerResponse_PB
