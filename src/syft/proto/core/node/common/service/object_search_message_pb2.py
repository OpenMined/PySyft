# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/service/object_search_message.proto
"""Generated protocol buffer code."""
# third party
from google.protobuf import descriptor as _descriptor
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
from syft.proto.core.pointer import (
    pointer_pb2 as proto_dot_core_dot_pointer_dot_pointer__pb2,
)

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/node/common/service/object_search_message.proto",
    package="syft.core.node.common.service",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n:proto/core/node/common/service/object_search_message.proto\x12\x1dsyft.core.node.common.service\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto\x1a proto/core/pointer/pointer.proto"\x8d\x01\n\x13ObjectSearchMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12\'\n\x08reply_to\x18\x03 \x01(\x0b\x32\x15.syft.core.io.Address"\x96\x01\n\x18ObjectSearchReplyMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x02 \x01(\x0b\x32\x15.syft.core.io.Address\x12+\n\x07results\x18\x03 \x03(\x0b\x32\x1a.syft.core.pointer.Pointerb\x06proto3',
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
        proto_dot_core_dot_io_dot_address__pb2.DESCRIPTOR,
        proto_dot_core_dot_pointer_dot_pointer__pb2.DESCRIPTOR,
    ],
)


_OBJECTSEARCHMESSAGE = _descriptor.Descriptor(
    name="ObjectSearchMessage",
    full_name="syft.core.node.common.service.ObjectSearchMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.core.node.common.service.ObjectSearchMessage.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.core.node.common.service.ObjectSearchMessage.address",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="reply_to",
            full_name="syft.core.node.common.service.ObjectSearchMessage.reply_to",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=196,
    serialized_end=337,
)


_OBJECTSEARCHREPLYMESSAGE = _descriptor.Descriptor(
    name="ObjectSearchReplyMessage",
    full_name="syft.core.node.common.service.ObjectSearchReplyMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.core.node.common.service.ObjectSearchReplyMessage.msg_id",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.core.node.common.service.ObjectSearchReplyMessage.address",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="results",
            full_name="syft.core.node.common.service.ObjectSearchReplyMessage.results",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=340,
    serialized_end=490,
)

_OBJECTSEARCHMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_OBJECTSEARCHMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_OBJECTSEARCHMESSAGE.fields_by_name[
    "reply_to"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_OBJECTSEARCHREPLYMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_OBJECTSEARCHREPLYMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_OBJECTSEARCHREPLYMESSAGE.fields_by_name[
    "results"
].message_type = proto_dot_core_dot_pointer_dot_pointer__pb2._POINTER
DESCRIPTOR.message_types_by_name["ObjectSearchMessage"] = _OBJECTSEARCHMESSAGE
DESCRIPTOR.message_types_by_name["ObjectSearchReplyMessage"] = _OBJECTSEARCHREPLYMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ObjectSearchMessage = _reflection.GeneratedProtocolMessageType(
    "ObjectSearchMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _OBJECTSEARCHMESSAGE,
        "__module__": "proto.core.node.common.service.object_search_message_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.node.common.service.ObjectSearchMessage)
    },
)
_sym_db.RegisterMessage(ObjectSearchMessage)

ObjectSearchReplyMessage = _reflection.GeneratedProtocolMessageType(
    "ObjectSearchReplyMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _OBJECTSEARCHREPLYMESSAGE,
        "__module__": "proto.core.node.common.service.object_search_message_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.node.common.service.ObjectSearchReplyMessage)
    },
)
_sym_db.RegisterMessage(ObjectSearchReplyMessage)


# @@protoc_insertion_point(module_scope)
