# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/io/location.proto

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

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/io/location.proto",
    package="syft.core.io",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x1cproto/core/io/location.proto\x12\x0csyft.core.io\x1a%proto/core/common/common_object.proto"C\n\x10SpecificLocation\x12!\n\x02id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3',
    dependencies=[proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,],
)


_SPECIFICLOCATION = _descriptor.Descriptor(
    name="SpecificLocation",
    full_name="syft.core.io.SpecificLocation",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="id",
            full_name="syft.core.io.SpecificLocation.id",
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
            name="name",
            full_name="syft.core.io.SpecificLocation.name",
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
    serialized_start=85,
    serialized_end=152,
)

_SPECIFICLOCATION.fields_by_name[
    "id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
DESCRIPTOR.message_types_by_name["SpecificLocation"] = _SPECIFICLOCATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SpecificLocation = _reflection.GeneratedProtocolMessageType(
    "SpecificLocation",
    (_message.Message,),
    {
        "DESCRIPTOR": _SPECIFICLOCATION,
        "__module__": "proto.core.io.location_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.io.SpecificLocation)
    },
)
_sym_db.RegisterMessage(SpecificLocation)


# @@protoc_insertion_point(module_scope)
