# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/grid/messages/infra_messages.proto
"""Generated protocol buffer code."""
# third party
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


# syft absolute
from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n(proto/grid/messages/infra_messages.proto\x12\x12syft.grid.messages\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto"\x9e\x01\n\x13\x43reateWorkerMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x8b\x01\n\x14\x43reateWorkerResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\xa8\x01\n\x1dGetWorkerInstanceTypesMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x95\x01\n\x1eGetWorkerInstanceTypesResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9b\x01\n\x10GetWorkerMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x88\x01\n\x11GetWorkerResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9c\x01\n\x11GetWorkersMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x89\x01\n\x12GetWorkersResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9e\x01\n\x13\x44\x65leteWorkerMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x8b\x01\n\x14\x44\x65leteWorkerResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x9e\x01\n\x13UpdateWorkerMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\'\n\x08reply_to\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address"\x8b\x01\n\x14UpdateWorkerResponse\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12&\n\x07\x61\x64\x64ress\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Addressb\x06proto3'
)


_CREATEWORKERMESSAGE = DESCRIPTOR.message_types_by_name["CreateWorkerMessage"]
_CREATEWORKERRESPONSE = DESCRIPTOR.message_types_by_name["CreateWorkerResponse"]
_GETWORKERINSTANCETYPESMESSAGE = DESCRIPTOR.message_types_by_name[
    "GetWorkerInstanceTypesMessage"
]
_GETWORKERINSTANCETYPESRESPONSE = DESCRIPTOR.message_types_by_name[
    "GetWorkerInstanceTypesResponse"
]
_GETWORKERMESSAGE = DESCRIPTOR.message_types_by_name["GetWorkerMessage"]
_GETWORKERRESPONSE = DESCRIPTOR.message_types_by_name["GetWorkerResponse"]
_GETWORKERSMESSAGE = DESCRIPTOR.message_types_by_name["GetWorkersMessage"]
_GETWORKERSRESPONSE = DESCRIPTOR.message_types_by_name["GetWorkersResponse"]
_DELETEWORKERMESSAGE = DESCRIPTOR.message_types_by_name["DeleteWorkerMessage"]
_DELETEWORKERRESPONSE = DESCRIPTOR.message_types_by_name["DeleteWorkerResponse"]
_UPDATEWORKERMESSAGE = DESCRIPTOR.message_types_by_name["UpdateWorkerMessage"]
_UPDATEWORKERRESPONSE = DESCRIPTOR.message_types_by_name["UpdateWorkerResponse"]
CreateWorkerMessage = _reflection.GeneratedProtocolMessageType(
    "CreateWorkerMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _CREATEWORKERMESSAGE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.CreateWorkerMessage)
    },
)
_sym_db.RegisterMessage(CreateWorkerMessage)

CreateWorkerResponse = _reflection.GeneratedProtocolMessageType(
    "CreateWorkerResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _CREATEWORKERRESPONSE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.CreateWorkerResponse)
    },
)
_sym_db.RegisterMessage(CreateWorkerResponse)

GetWorkerInstanceTypesMessage = _reflection.GeneratedProtocolMessageType(
    "GetWorkerInstanceTypesMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _GETWORKERINSTANCETYPESMESSAGE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetWorkerInstanceTypesMessage)
    },
)
_sym_db.RegisterMessage(GetWorkerInstanceTypesMessage)

GetWorkerInstanceTypesResponse = _reflection.GeneratedProtocolMessageType(
    "GetWorkerInstanceTypesResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _GETWORKERINSTANCETYPESRESPONSE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetWorkerInstanceTypesResponse)
    },
)
_sym_db.RegisterMessage(GetWorkerInstanceTypesResponse)

GetWorkerMessage = _reflection.GeneratedProtocolMessageType(
    "GetWorkerMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _GETWORKERMESSAGE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetWorkerMessage)
    },
)
_sym_db.RegisterMessage(GetWorkerMessage)

GetWorkerResponse = _reflection.GeneratedProtocolMessageType(
    "GetWorkerResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _GETWORKERRESPONSE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetWorkerResponse)
    },
)
_sym_db.RegisterMessage(GetWorkerResponse)

GetWorkersMessage = _reflection.GeneratedProtocolMessageType(
    "GetWorkersMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _GETWORKERSMESSAGE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetWorkersMessage)
    },
)
_sym_db.RegisterMessage(GetWorkersMessage)

GetWorkersResponse = _reflection.GeneratedProtocolMessageType(
    "GetWorkersResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _GETWORKERSRESPONSE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.GetWorkersResponse)
    },
)
_sym_db.RegisterMessage(GetWorkersResponse)

DeleteWorkerMessage = _reflection.GeneratedProtocolMessageType(
    "DeleteWorkerMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _DELETEWORKERMESSAGE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.DeleteWorkerMessage)
    },
)
_sym_db.RegisterMessage(DeleteWorkerMessage)

DeleteWorkerResponse = _reflection.GeneratedProtocolMessageType(
    "DeleteWorkerResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _DELETEWORKERRESPONSE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.DeleteWorkerResponse)
    },
)
_sym_db.RegisterMessage(DeleteWorkerResponse)

UpdateWorkerMessage = _reflection.GeneratedProtocolMessageType(
    "UpdateWorkerMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _UPDATEWORKERMESSAGE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.UpdateWorkerMessage)
    },
)
_sym_db.RegisterMessage(UpdateWorkerMessage)

UpdateWorkerResponse = _reflection.GeneratedProtocolMessageType(
    "UpdateWorkerResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _UPDATEWORKERRESPONSE,
        "__module__": "proto.grid.messages.infra_messages_pb2"
        # @@protoc_insertion_point(class_scope:syft.grid.messages.UpdateWorkerResponse)
    },
)
_sym_db.RegisterMessage(UpdateWorkerResponse)

if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _CREATEWORKERMESSAGE._serialized_start = 133
    _CREATEWORKERMESSAGE._serialized_end = 291
    _CREATEWORKERRESPONSE._serialized_start = 294
    _CREATEWORKERRESPONSE._serialized_end = 433
    _GETWORKERINSTANCETYPESMESSAGE._serialized_start = 436
    _GETWORKERINSTANCETYPESMESSAGE._serialized_end = 604
    _GETWORKERINSTANCETYPESRESPONSE._serialized_start = 607
    _GETWORKERINSTANCETYPESRESPONSE._serialized_end = 756
    _GETWORKERMESSAGE._serialized_start = 759
    _GETWORKERMESSAGE._serialized_end = 914
    _GETWORKERRESPONSE._serialized_start = 917
    _GETWORKERRESPONSE._serialized_end = 1053
    _GETWORKERSMESSAGE._serialized_start = 1056
    _GETWORKERSMESSAGE._serialized_end = 1212
    _GETWORKERSRESPONSE._serialized_start = 1215
    _GETWORKERSRESPONSE._serialized_end = 1352
    _DELETEWORKERMESSAGE._serialized_start = 1355
    _DELETEWORKERMESSAGE._serialized_end = 1513
    _DELETEWORKERRESPONSE._serialized_start = 1516
    _DELETEWORKERRESPONSE._serialized_end = 1655
    _UPDATEWORKERMESSAGE._serialized_start = 1658
    _UPDATEWORKERMESSAGE._serialized_end = 1816
    _UPDATEWORKERRESPONSE._serialized_start = 1819
    _UPDATEWORKERRESPONSE._serialized_end = 1958
# @@protoc_insertion_point(module_scope)
