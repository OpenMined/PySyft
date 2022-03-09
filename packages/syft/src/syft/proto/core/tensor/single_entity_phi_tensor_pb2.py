# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/tensor/single_entity_phi_tensor.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.adp import entity_pb2 as proto_dot_core_dot_adp_dot_entity__pb2
from syft.proto.lib.numpy import array_pb2 as proto_dot_lib_dot_numpy_dot_array__pb2
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2
from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b"\n0proto/core/tensor/single_entity_phi_tensor.proto\x12\x10syft.core.tensor\x1a\x1bproto/core/adp/entity.proto\x1a\x1bproto/lib/numpy/array.proto\x1a\x1bproto/core/io/address.proto\x1a%proto/core/common/common_object.proto\"\x82\x03\n)TensorWrappedSingleEntityPhiTensorPointer\x12%\n\x06\x65ntity\x18\x01 \x01(\x0b\x32\x15.syft.core.adp.Entity\x12,\n\x08min_vals\x18\x02 \x01(\x0b\x32\x1a.syft.lib.numpy.NumpyProto\x12,\n\x08max_vals\x18\x03 \x01(\x0b\x32\x1a.syft.lib.numpy.NumpyProto\x12'\n\x08location\x18\x04 \x01(\x0b\x32\x15.syft.core.io.Address\x12\x16\n\x0escalar_manager\x18\x05 \x01(\x0c\x12-\n\x0eid_at_location\x18\x06 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x13\n\x0bobject_type\x18\x07 \x01(\t\x12\x0c\n\x04tags\x18\x08 \x03(\t\x12\x13\n\x0b\x64\x65scription\x18\t \x01(\t\x12\x14\n\x0cpublic_shape\x18\n \x01(\x0c\x12\x14\n\x0cpublic_dtype\x18\x0b \x01(\x0c\x62\x06proto3"
)


_TENSORWRAPPEDSINGLEENTITYPHITENSORPOINTER = DESCRIPTOR.message_types_by_name[
    "TensorWrappedSingleEntityPhiTensorPointer"
]
TensorWrappedSingleEntityPhiTensorPointer = _reflection.GeneratedProtocolMessageType(
    "TensorWrappedSingleEntityPhiTensorPointer",
    (_message.Message,),
    {
        "DESCRIPTOR": _TENSORWRAPPEDSINGLEENTITYPHITENSORPOINTER,
        "__module__": "proto.core.tensor.single_entity_phi_tensor_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.tensor.TensorWrappedSingleEntityPhiTensorPointer)
    },
)
_sym_db.RegisterMessage(TensorWrappedSingleEntityPhiTensorPointer)

if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _TENSORWRAPPEDSINGLEENTITYPHITENSORPOINTER._serialized_start = 197
    _TENSORWRAPPEDSINGLEENTITYPHITENSORPOINTER._serialized_end = 583
# @@protoc_insertion_point(module_scope)
