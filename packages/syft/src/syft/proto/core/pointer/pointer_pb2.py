# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/pointer/pointer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n proto/core/pointer/pointer.proto\x12\x11syft.core.pointer\x1a%proto/core/common/common_object.proto\x1a\x1bproto/core/io/address.proto"\xcd\x02\n\x07Pointer\x12"\n\x1apoints_to_object_with_path\x18\x01 \x01(\t\x12\x14\n\x0cpointer_name\x18\x02 \x01(\t\x12-\n\x0eid_at_location\x18\x03 \x01(\x0b\x32\x15.syft.core.common.UID\x12\'\n\x08location\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x0c\n\x04tags\x18\x05 \x03(\t\x12\x13\n\x0b\x64\x65scription\x18\x06 \x01(\t\x12\x13\n\x0bobject_type\x18\x07 \x01(\t\x12\x16\n\x0e\x61ttribute_name\x18\x08 \x01(\t\x12\x19\n\x0cpublic_shape\x18\t \x01(\x0cH\x00\x88\x01\x01\x12\x1e\n\x11obj_public_kwargs\x18\n \x01(\x0cH\x01\x88\x01\x01\x42\x0f\n\r_public_shapeB\x14\n\x12_obj_public_kwargsb\x06proto3'
)


_POINTER = DESCRIPTOR.message_types_by_name["Pointer"]
Pointer = _reflection.GeneratedProtocolMessageType(
    "Pointer",
    (_message.Message,),
    {
        "DESCRIPTOR": _POINTER,
        "__module__": "proto.core.pointer.pointer_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.pointer.Pointer)
    },
)
_sym_db.RegisterMessage(Pointer)

if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _POINTER._serialized_start = 124
    _POINTER._serialized_end = 457
# @@protoc_insertion_point(module_scope)
