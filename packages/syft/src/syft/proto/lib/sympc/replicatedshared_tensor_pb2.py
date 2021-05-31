# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/lib/sympc/replicatedshared_tensor.proto
"""Generated protocol buffer code."""
# third party
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


# syft absolute
from syft.proto.lib.sympc import session_pb2 as proto_dot_lib_dot_sympc_dot_session__pb2
from syft.proto.lib.torch import tensor_pb2 as proto_dot_lib_dot_torch_dot_tensor__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/lib/sympc/replicatedshared_tensor.proto",
    package="syft.lib.sympc",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n-proto/lib/sympc/replicatedshared_tensor.proto\x12\x0esyft.lib.sympc\x1a\x1cproto/lib/torch/tensor.proto\x1a\x1dproto/lib/sympc/session.proto"q\n\x16ReplicatedSharedTensor\x12*\n\x06tensor\x18\x01 \x03(\x0b\x32\x1a.syft.lib.torch.TensorData\x12+\n\x07session\x18\x02 \x01(\x0b\x32\x1a.syft.lib.sympc.MPCSessionb\x06proto3',
    dependencies=[
        proto_dot_lib_dot_torch_dot_tensor__pb2.DESCRIPTOR,
        proto_dot_lib_dot_sympc_dot_session__pb2.DESCRIPTOR,
    ],
)


_REPLICATEDSHAREDTENSOR = _descriptor.Descriptor(
    name="ReplicatedSharedTensor",
    full_name="syft.lib.sympc.ReplicatedSharedTensor",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="tensor",
            full_name="syft.lib.sympc.ReplicatedSharedTensor.tensor",
            index=0,
            number=1,
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
        _descriptor.FieldDescriptor(
            name="session",
            full_name="syft.lib.sympc.ReplicatedSharedTensor.session",
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
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=126,
    serialized_end=239,
)

_REPLICATEDSHAREDTENSOR.fields_by_name[
    "tensor"
].message_type = proto_dot_lib_dot_torch_dot_tensor__pb2._TENSORDATA
_REPLICATEDSHAREDTENSOR.fields_by_name[
    "session"
].message_type = proto_dot_lib_dot_sympc_dot_session__pb2._MPCSESSION
DESCRIPTOR.message_types_by_name["ReplicatedSharedTensor"] = _REPLICATEDSHAREDTENSOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ReplicatedSharedTensor = _reflection.GeneratedProtocolMessageType(
    "ReplicatedSharedTensor",
    (_message.Message,),
    {
        "DESCRIPTOR": _REPLICATEDSHAREDTENSOR,
        "__module__": "proto.lib.sympc.replicatedshared_tensor_pb2"
        # @@protoc_insertion_point(class_scope:syft.lib.sympc.ReplicatedSharedTensor)
    },
)
_sym_db.RegisterMessage(ReplicatedSharedTensor)


# @@protoc_insertion_point(module_scope)
