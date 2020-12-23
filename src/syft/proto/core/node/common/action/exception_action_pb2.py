# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/action/exception_action.proto

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

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/node/common/action/exception_action.proto",
    package="syft.core.node.common.service",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n4proto/core/node/common/action/exception_action.proto\x12\x1dsyft.core.node.common.service\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto"\xa2\x01\n\x10\x45xceptionMessage\x12&\n\x07\x61\x64\x64ress\x18\x01 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x37\n\x18msg_id_causing_exception\x18\x02 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x16\n\x0e\x65xception_type\x18\x03 \x01(\t\x12\x15\n\rexception_msg\x18\x04 \x01(\tb\x06proto3',
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
        proto_dot_core_dot_io_dot_address__pb2.DESCRIPTOR,
    ],
)


_EXCEPTIONMESSAGE = _descriptor.Descriptor(
    name="ExceptionMessage",
    full_name="syft.core.node.common.service.ExceptionMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="address",
            full_name="syft.core.node.common.service.ExceptionMessage.address",
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
            name="msg_id_causing_exception",
            full_name="syft.core.node.common.service.ExceptionMessage.msg_id_causing_exception",
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
            name="exception_type",
            full_name="syft.core.node.common.service.ExceptionMessage.exception_type",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
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
            name="exception_msg",
            full_name="syft.core.node.common.service.ExceptionMessage.exception_msg",
            index=3,
            number=4,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
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
    serialized_start=156,
    serialized_end=318,
)

_EXCEPTIONMESSAGE.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_EXCEPTIONMESSAGE.fields_by_name[
    "msg_id_causing_exception"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
DESCRIPTOR.message_types_by_name["ExceptionMessage"] = _EXCEPTIONMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ExceptionMessage = _reflection.GeneratedProtocolMessageType(
    "ExceptionMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _EXCEPTIONMESSAGE,
        "__module__": "proto.core.node.common.action.exception_action_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.node.common.service.ExceptionMessage)
    },
)
_sym_db.RegisterMessage(ExceptionMessage)


# @@protoc_insertion_point(module_scope)
