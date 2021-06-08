# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/lib/numpy/array.proto
"""Generated protocol buffer code."""
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
    name="proto/lib/numpy/array.proto",
    package="syft.lib.numpy",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x1bproto/lib/numpy/array.proto\x12\x0esyft.lib.numpy\x1a\x1cproto/lib/torch/tensor.proto"O\n\nNumpyProto\x12\x14\n\narrow_data\x18\x01 \x01(\x0cH\x00\x12\x14\n\nproto_data\x18\x02 \x01(\x0cH\x00\x12\r\n\x05\x64type\x18\x03 \x01(\tB\x06\n\x04\x64\x61tab\x06proto3',
    dependencies=[
        proto_dot_lib_dot_torch_dot_tensor__pb2.DESCRIPTOR,
    ],
)


_NUMPYPROTO = _descriptor.Descriptor(
    name="NumpyProto",
    full_name="syft.lib.numpy.NumpyProto",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="arrow_data",
            full_name="syft.lib.numpy.NumpyProto.arrow_data",
            index=0,
            number=1,
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="proto_data",
            full_name="syft.lib.numpy.NumpyProto.proto_data",
            index=1,
            number=2,
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
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="dtype",
            full_name="syft.lib.numpy.NumpyProto.dtype",
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
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name="data",
            full_name="syft.lib.numpy.NumpyProto.data",
            index=0,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
    ],
    serialized_start=77,
    serialized_end=156,
)

_NUMPYPROTO.oneofs_by_name["data"].fields.append(
    _NUMPYPROTO.fields_by_name["arrow_data"]
)
_NUMPYPROTO.fields_by_name["arrow_data"].containing_oneof = _NUMPYPROTO.oneofs_by_name[
    "data"
]
_NUMPYPROTO.oneofs_by_name["data"].fields.append(
    _NUMPYPROTO.fields_by_name["proto_data"]
)
_NUMPYPROTO.fields_by_name["proto_data"].containing_oneof = _NUMPYPROTO.oneofs_by_name[
    "data"
]
DESCRIPTOR.message_types_by_name["NumpyProto"] = _NUMPYPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NumpyProto = _reflection.GeneratedProtocolMessageType(
    "NumpyProto",
    (_message.Message,),
    {
        "DESCRIPTOR": _NUMPYPROTO,
        "__module__": "proto.lib.numpy.array_pb2"
        # @@protoc_insertion_point(class_scope:syft.lib.numpy.NumpyProto)
    },
)
_sym_db.RegisterMessage(NumpyProto)


# @@protoc_insertion_point(module_scope)
