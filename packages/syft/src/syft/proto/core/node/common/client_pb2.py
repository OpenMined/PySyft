# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/client.proto
"""Generated protocol buffer code."""
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
from syft.proto.core.io import location_pb2 as proto_dot_core_dot_io_dot_location__pb2
from syft.proto.core.io import route_pb2 as proto_dot_core_dot_io_dot_route__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/node/common/client.proto",
    package="syft.core.node.common",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b"\n#proto/core/node/common/client.proto\x12\x15syft.core.node.common\x1a%proto/core/common/common_object.proto\x1a\x1cproto/core/io/location.proto\x1a\x19proto/core/io/route.proto\"\xee\x02\n\x06\x43lient\x12!\n\x02id\x18\x01 \x01(\x0b\x32\x15.syft.core.common.UID\x12\x0c\n\x04name\x18\x02 \x01(\t\x12'\n\x06routes\x18\x03 \x03(\x0b\x32\x17.syft.core.io.SoloRoute\x12\x34\n\x07network\x18\x04 \x01(\x0b\x32\x1e.syft.core.io.SpecificLocationH\x00\x88\x01\x01\x12\x33\n\x06\x64omain\x18\x05 \x01(\x0b\x32\x1e.syft.core.io.SpecificLocationH\x01\x88\x01\x01\x12\x33\n\x06\x64\x65vice\x18\x06 \x01(\x0b\x32\x1e.syft.core.io.SpecificLocationH\x02\x88\x01\x01\x12/\n\x02vm\x18\x07 \x01(\x0b\x32\x1e.syft.core.io.SpecificLocationH\x03\x88\x01\x01\x12\x10\n\x08obj_type\x18\x08 \x01(\tB\n\n\x08_networkB\t\n\x07_domainB\t\n\x07_deviceB\x05\n\x03_vmb\x06proto3",
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
        proto_dot_core_dot_io_dot_location__pb2.DESCRIPTOR,
        proto_dot_core_dot_io_dot_route__pb2.DESCRIPTOR,
    ],
)


_CLIENT = _descriptor.Descriptor(
    name="Client",
    full_name="syft.core.node.common.Client",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="id",
            full_name="syft.core.node.common.Client.id",
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
            full_name="syft.core.node.common.Client.name",
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
        _descriptor.FieldDescriptor(
            name="routes",
            full_name="syft.core.node.common.Client.routes",
            index=2,
            number=3,
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
            name="network",
            full_name="syft.core.node.common.Client.network",
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
        _descriptor.FieldDescriptor(
            name="domain",
            full_name="syft.core.node.common.Client.domain",
            index=4,
            number=5,
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
            name="device",
            full_name="syft.core.node.common.Client.device",
            index=5,
            number=6,
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
            name="vm",
            full_name="syft.core.node.common.Client.vm",
            index=6,
            number=7,
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
            name="obj_type",
            full_name="syft.core.node.common.Client.obj_type",
            index=7,
            number=8,
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
            name="_network",
            full_name="syft.core.node.common.Client._network",
            index=0,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
        _descriptor.OneofDescriptor(
            name="_domain",
            full_name="syft.core.node.common.Client._domain",
            index=1,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
        _descriptor.OneofDescriptor(
            name="_device",
            full_name="syft.core.node.common.Client._device",
            index=2,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
        _descriptor.OneofDescriptor(
            name="_vm",
            full_name="syft.core.node.common.Client._vm",
            index=3,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
    ],
    serialized_start=159,
    serialized_end=525,
)

_CLIENT.fields_by_name[
    "id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_CLIENT.fields_by_name[
    "routes"
].message_type = proto_dot_core_dot_io_dot_route__pb2._SOLOROUTE
_CLIENT.fields_by_name[
    "network"
].message_type = proto_dot_core_dot_io_dot_location__pb2._SPECIFICLOCATION
_CLIENT.fields_by_name[
    "domain"
].message_type = proto_dot_core_dot_io_dot_location__pb2._SPECIFICLOCATION
_CLIENT.fields_by_name[
    "device"
].message_type = proto_dot_core_dot_io_dot_location__pb2._SPECIFICLOCATION
_CLIENT.fields_by_name[
    "vm"
].message_type = proto_dot_core_dot_io_dot_location__pb2._SPECIFICLOCATION
_CLIENT.oneofs_by_name["_network"].fields.append(_CLIENT.fields_by_name["network"])
_CLIENT.fields_by_name["network"].containing_oneof = _CLIENT.oneofs_by_name["_network"]
_CLIENT.oneofs_by_name["_domain"].fields.append(_CLIENT.fields_by_name["domain"])
_CLIENT.fields_by_name["domain"].containing_oneof = _CLIENT.oneofs_by_name["_domain"]
_CLIENT.oneofs_by_name["_device"].fields.append(_CLIENT.fields_by_name["device"])
_CLIENT.fields_by_name["device"].containing_oneof = _CLIENT.oneofs_by_name["_device"]
_CLIENT.oneofs_by_name["_vm"].fields.append(_CLIENT.fields_by_name["vm"])
_CLIENT.fields_by_name["vm"].containing_oneof = _CLIENT.oneofs_by_name["_vm"]
DESCRIPTOR.message_types_by_name["Client"] = _CLIENT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Client = _reflection.GeneratedProtocolMessageType(
    "Client",
    (_message.Message,),
    {
        "DESCRIPTOR": _CLIENT,
        "__module__": "proto.core.node.common.client_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.node.common.Client)
    },
)
_sym_db.RegisterMessage(Client)


# @@protoc_insertion_point(module_scope)
