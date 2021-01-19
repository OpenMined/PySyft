# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/lib/torch/parameter.proto

# third party
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


# syft absolute
from syft.proto.lib.torch import tensor_pb2 as proto_dot_lib_dot_torch_dot_tensor__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/lib/torch/parameter.proto",
    package="syft.lib.torch",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x1fproto/lib/torch/parameter.proto\x12\x0esyft.lib.torch\x1a\x1cproto/lib/torch/tensor.proto"\xae\x01\n\x0eParameterProto\x12*\n\x06tensor\x18\x01 \x01(\x0b\x32\x1a.syft.lib.torch.TensorData\x12\x15\n\rrequires_grad\x18\x02 \x01(\x08\x12(\n\x04grad\x18\x03 \x01(\x0b\x32\x1a.syft.lib.torch.TensorData\x12/\n\x0bgrad_sample\x18\x04 \x01(\x0b\x32\x1a.syft.lib.torch.TensorDatab\x06proto3',
    dependencies=[
        proto_dot_lib_dot_torch_dot_tensor__pb2.DESCRIPTOR,
    ],
)


_PARAMETERPROTO = _descriptor.Descriptor(
    name="ParameterProto",
    full_name="syft.lib.torch.ParameterProto",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="tensor",
            full_name="syft.lib.torch.ParameterProto.tensor",
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
            name="requires_grad",
            full_name="syft.lib.torch.ParameterProto.requires_grad",
            index=1,
            number=2,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
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
            name="grad",
            full_name="syft.lib.torch.ParameterProto.grad",
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
        _descriptor.FieldDescriptor(
            name="grad_sample",
            full_name="syft.lib.torch.ParameterProto.grad_sample",
            index=3,
            number=4,
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
    serialized_start=82,
    serialized_end=256,
)

_PARAMETERPROTO.fields_by_name[
    "tensor"
].message_type = proto_dot_lib_dot_torch_dot_tensor__pb2._TENSORDATA
_PARAMETERPROTO.fields_by_name[
    "grad"
].message_type = proto_dot_lib_dot_torch_dot_tensor__pb2._TENSORDATA
_PARAMETERPROTO.fields_by_name[
    "grad_sample"
].message_type = proto_dot_lib_dot_torch_dot_tensor__pb2._TENSORDATA
DESCRIPTOR.message_types_by_name["ParameterProto"] = _PARAMETERPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ParameterProto = _reflection.GeneratedProtocolMessageType(
    "ParameterProto",
    (_message.Message,),
    {
        "DESCRIPTOR": _PARAMETERPROTO,
        "__module__": "proto.lib.torch.parameter_pb2"
        # @@protoc_insertion_point(class_scope:syft.lib.torch.ParameterProto)
    },
)
_sym_db.RegisterMessage(ParameterProto)


# @@protoc_insertion_point(module_scope)
