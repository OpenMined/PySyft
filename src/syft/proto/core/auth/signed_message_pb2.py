# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/auth/signed_message.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)


DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/auth/signed_message.proto",
    package="syft.core.auth",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=b'\n$proto/core/auth/signed_message.proto\x12\x0esyft.core.auth\x1a%proto/core/common/common_object.proto"\x80\x01\n\rSignedMessage\x12%\n\x06msg_id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x10\n\x08obj_type\x18\x02 \x01(\t\x12\x11\n\tsignature\x18\x03 \x01(\x0c\x12\x12\n\nverify_key\x18\x04 \x01(\x0c\x12\x0f\n\x07message\x18\x05 \x01(\x0c\x62\x06proto3',
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
    ],
)


_SIGNEDMESSAGE = _descriptor.Descriptor(
    name="SignedMessage",
    full_name="syft.core.auth.SignedMessage",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="msg_id",
            full_name="syft.core.auth.SignedMessage.msg_id",
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
        ),
        _descriptor.FieldDescriptor(
            name="obj_type",
            full_name="syft.core.auth.SignedMessage.obj_type",
            index=1,
            number=2,
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
        ),
        _descriptor.FieldDescriptor(
            name="signature",
            full_name="syft.core.auth.SignedMessage.signature",
            index=2,
            number=3,
            type=12,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"",
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="verify_key",
            full_name="syft.core.auth.SignedMessage.verify_key",
            index=3,
            number=4,
            type=12,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"",
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="message",
            full_name="syft.core.auth.SignedMessage.message",
            index=4,
            number=5,
            type=12,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"",
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
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
    serialized_start=96,
    serialized_end=224,
)

_SIGNEDMESSAGE.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
DESCRIPTOR.message_types_by_name["SignedMessage"] = _SIGNEDMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SignedMessage = _reflection.GeneratedProtocolMessageType(
    "SignedMessage",
    (_message.Message,),
    {
        "DESCRIPTOR": _SIGNEDMESSAGE,
        "__module__": "proto.core.auth.signed_message_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.auth.SignedMessage)
    },
)
_sym_db.RegisterMessage(SignedMessage)


# @@protoc_insertion_point(module_scope)
