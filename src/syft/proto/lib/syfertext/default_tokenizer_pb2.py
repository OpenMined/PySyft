# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/lib/syfertext/default_tokenizer.proto

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
    name="proto/lib/syfertext/default_tokenizer.proto",
    package="syft.lib.syfertext",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n+proto/lib/syfertext/default_tokenizer.proto\x12\x12syft.lib.syfertext\x1a%proto/core/common/common_object.proto"l\n\x10\x44\x65\x66\x61ultTokenizer\x12#\n\x04uuid\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x10\n\x08prefixes\x18\x02 \x03(\t\x12\x10\n\x08suffixes\x18\x03 \x03(\t\x12\x0f\n\x07infixes\x18\x04 \x03(\tb\x06proto3',
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
    ],
)


_DEFAULTTOKENIZER = _descriptor.Descriptor(
    name="DefaultTokenizer",
    full_name="syft.lib.syfertext.DefaultTokenizer",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="uuid",
            full_name="syft.lib.syfertext.DefaultTokenizer.uuid",
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
            name="prefixes",
            full_name="syft.lib.syfertext.DefaultTokenizer.prefixes",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
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
        _descriptor.FieldDescriptor(
            name="suffixes",
            full_name="syft.lib.syfertext.DefaultTokenizer.suffixes",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
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
        _descriptor.FieldDescriptor(
            name="infixes",
            full_name="syft.lib.syfertext.DefaultTokenizer.infixes",
            index=3,
            number=4,
            type=9,
            cpp_type=9,
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
    serialized_start=106,
    serialized_end=214,
)

_DEFAULTTOKENIZER.fields_by_name[
    "uuid"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
DESCRIPTOR.message_types_by_name["DefaultTokenizer"] = _DEFAULTTOKENIZER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DefaultTokenizer = _reflection.GeneratedProtocolMessageType(
    "DefaultTokenizer",
    (_message.Message,),
    {
        "DESCRIPTOR": _DEFAULTTOKENIZER,
        "__module__": "proto.lib.syfertext.default_tokenizer_pb2"
        # @@protoc_insertion_point(class_scope:syft.lib.syfertext.DefaultTokenizer)
    },
)
_sym_db.RegisterMessage(DefaultTokenizer)


# @@protoc_insertion_point(module_scope)
