# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/sample.proto

# stdlib
import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
# third party
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


# syft absolute
# absolute
from syft.proto.lib.python import string_pb2 as proto_dot_lib_dot_python_dot_string__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/sample.proto",
    package="numpy",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=_b(
        '\n\x12proto/sample.proto\x12\x05numpy\x1a\x1dproto/lib/python/string.proto",\n\x03Msg\x12%\n\x04name\x18\x01 \x01(\x0b\x32\x17.syft.lib.python.Stringb\x06proto3'
    ),
    dependencies=[
        proto_dot_lib_dot_python_dot_string__pb2.DESCRIPTOR,
    ],
)


_MSG = _descriptor.Descriptor(
    name="Msg",
    full_name="numpy.Msg",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="name",
            full_name="numpy.Msg.name",
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
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=60,
    serialized_end=104,
)

_MSG.fields_by_name[
    "name"
].message_type = proto_dot_lib_dot_python_dot_string__pb2._STRING
DESCRIPTOR.message_types_by_name["Msg"] = _MSG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Msg = _reflection.GeneratedProtocolMessageType(
    "Msg",
    (_message.Message,),
    dict(
        DESCRIPTOR=_MSG,
        __module__="proto.sample_pb2"
        # @@protoc_insertion_point(class_scope:numpy.Msg)
    ),
)
_sym_db.RegisterMessage(Msg)


# @@protoc_insertion_point(module_scope)
