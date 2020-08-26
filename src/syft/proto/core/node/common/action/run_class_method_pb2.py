# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/node/common/action/run_class_method.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft.proto.core.common import (
    common_object_pb2 as proto_dot_core_dot_common_dot_common__object__pb2,
)
from syft.proto.core.pointer import (
    pointer_pb2 as proto_dot_core_dot_pointer_dot_pointer__pb2,
)
from syft.proto.core.io import address_pb2 as proto_dot_core_dot_io_dot_address__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/node/common/action/run_class_method.proto",
    package="syft.core.node.common.action",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n4proto/core/node/common/action/run_class_method.proto\x12\x1csyft.core.node.common.action\x1a%proto/core/common/common_object.proto\x1a proto/core/pointer/pointer.proto\x1a\x1bproto/core/io/address.proto"\x92\x03\n\x14RunClassMethodAction\x12\x0c\n\x04path\x18\x01 \x01(\t\x12)\n\x05_self\x18\x02 \x01(\x0b\x32\x1a.syft.core.pointer.Pointer\x12(\n\x04\x61rgs\x18\x03 \x03(\x0b\x32\x1a.syft.core.pointer.Pointer\x12N\n\x06kwargs\x18\x04 \x03(\x0b\x32>.syft.core.node.common.action.RunClassMethodAction.KwargsEntry\x12-\n\x0eid_at_location\x18\x05 \x01(\x0b\x32\x15.syft.core.common.UID\x12&\n\x07\x61\x64\x64ress\x18\x06 \x01(\x0b\x32\x15.syft.core.io.Address\x12%\n\x06msg_id\x18\x07 \x01(\x0b\x32\x15.syft.core.common.UID\x1aI\n\x0bKwargsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12)\n\x05value\x18\x02 \x01(\x0b\x32\x1a.syft.core.pointer.Pointer:\x02\x38\x01\x62\x06proto3',
    dependencies=[
        proto_dot_core_dot_common_dot_common__object__pb2.DESCRIPTOR,
        proto_dot_core_dot_pointer_dot_pointer__pb2.DESCRIPTOR,
        proto_dot_core_dot_io_dot_address__pb2.DESCRIPTOR,
    ],
)


_RUNCLASSMETHODACTION_KWARGSENTRY = _descriptor.Descriptor(
    name="KwargsEntry",
    full_name="syft.core.node.common.action.RunClassMethodAction.KwargsEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="syft.core.node.common.action.RunClassMethodAction.KwargsEntry.key",
            index=0,
            number=1,
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
            name="value",
            full_name="syft.core.node.common.action.RunClassMethodAction.KwargsEntry.value",
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
    serialized_options=b"8\001",
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=518,
    serialized_end=591,
)

_RUNCLASSMETHODACTION = _descriptor.Descriptor(
    name="RunClassMethodAction",
    full_name="syft.core.node.common.action.RunClassMethodAction",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="path",
            full_name="syft.core.node.common.action.RunClassMethodAction.path",
            index=0,
            number=1,
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
            name="_self",
            full_name="syft.core.node.common.action.RunClassMethodAction._self",
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
        _descriptor.FieldDescriptor(
            name="args",
            full_name="syft.core.node.common.action.RunClassMethodAction.args",
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
            name="kwargs",
            full_name="syft.core.node.common.action.RunClassMethodAction.kwargs",
            index=3,
            number=4,
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
            name="id_at_location",
            full_name="syft.core.node.common.action.RunClassMethodAction.id_at_location",
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
            name="address",
            full_name="syft.core.node.common.action.RunClassMethodAction.address",
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
            name="msg_id",
            full_name="syft.core.node.common.action.RunClassMethodAction.msg_id",
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
    ],
    extensions=[],
    nested_types=[_RUNCLASSMETHODACTION_KWARGSENTRY,],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=189,
    serialized_end=591,
)

_RUNCLASSMETHODACTION_KWARGSENTRY.fields_by_name[
    "value"
].message_type = proto_dot_core_dot_pointer_dot_pointer__pb2._POINTER
_RUNCLASSMETHODACTION_KWARGSENTRY.containing_type = _RUNCLASSMETHODACTION
_RUNCLASSMETHODACTION.fields_by_name[
    "_self"
].message_type = proto_dot_core_dot_pointer_dot_pointer__pb2._POINTER
_RUNCLASSMETHODACTION.fields_by_name[
    "args"
].message_type = proto_dot_core_dot_pointer_dot_pointer__pb2._POINTER
_RUNCLASSMETHODACTION.fields_by_name[
    "kwargs"
].message_type = _RUNCLASSMETHODACTION_KWARGSENTRY
_RUNCLASSMETHODACTION.fields_by_name[
    "id_at_location"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
_RUNCLASSMETHODACTION.fields_by_name[
    "address"
].message_type = proto_dot_core_dot_io_dot_address__pb2._ADDRESS
_RUNCLASSMETHODACTION.fields_by_name[
    "msg_id"
].message_type = proto_dot_core_dot_common_dot_common__object__pb2._UID
DESCRIPTOR.message_types_by_name["RunClassMethodAction"] = _RUNCLASSMETHODACTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RunClassMethodAction = _reflection.GeneratedProtocolMessageType(
    "RunClassMethodAction",
    (_message.Message,),
    {
        "KwargsEntry": _reflection.GeneratedProtocolMessageType(
            "KwargsEntry",
            (_message.Message,),
            {
                "DESCRIPTOR": _RUNCLASSMETHODACTION_KWARGSENTRY,
                "__module__": "proto.core.node.common.action.run_class_method_pb2"
                # @@protoc_insertion_point(class_scope:syft.core.node.common.action.RunClassMethodAction.KwargsEntry)
            },
        ),
        "DESCRIPTOR": _RUNCLASSMETHODACTION,
        "__module__": "proto.core.node.common.action.run_class_method_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.node.common.action.RunClassMethodAction)
    },
)
_sym_db.RegisterMessage(RunClassMethodAction)
_sym_db.RegisterMessage(RunClassMethodAction.KwargsEntry)


_RUNCLASSMETHODACTION_KWARGSENTRY._options = None
# @@protoc_insertion_point(module_scope)
