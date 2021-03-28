# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/core/plan/plan.proto

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
from syft.proto.core.node.common.action import (
    action_pb2 as proto_dot_core_dot_node_dot_common_dot_action_dot_action__pb2,
)
from syft.proto.core.pointer import (
    pointer_pb2 as proto_dot_core_dot_pointer_dot_pointer__pb2,
)

DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/core/plan/plan.proto",
    package="syft.core.plan",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=_b(
        '\n\x1aproto/core/plan/plan.proto\x12\x0esyft.core.plan\x1a*proto/core/node/common/action/action.proto\x1a proto/core/pointer/pointer.proto"\xe7\x01\n\x04Plan\x12\x35\n\x07\x61\x63tions\x18\x01 \x03(\x0b\x32$.syft.core.node.common.action.Action\x12\x30\n\x06inputs\x18\x02 \x03(\x0b\x32 .syft.core.plan.Plan.InputsEntry\x12+\n\x07outputs\x18\x03 \x03(\x0b\x32\x1a.syft.core.pointer.Pointer\x1aI\n\x0bInputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12)\n\x05value\x18\x02 \x01(\x0b\x32\x1a.syft.core.pointer.Pointer:\x02\x38\x01\x62\x06proto3'
    ),
    dependencies=[
        proto_dot_core_dot_node_dot_common_dot_action_dot_action__pb2.DESCRIPTOR,
        proto_dot_core_dot_pointer_dot_pointer__pb2.DESCRIPTOR,
    ],
)


_PLAN_INPUTSENTRY = _descriptor.Descriptor(
    name="InputsEntry",
    full_name="syft.core.plan.Plan.InputsEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="syft.core.plan.Plan.InputsEntry.key",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="value",
            full_name="syft.core.plan.Plan.InputsEntry.value",
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
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=_b("8\001"),
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=283,
    serialized_end=356,
)

_PLAN = _descriptor.Descriptor(
    name="Plan",
    full_name="syft.core.plan.Plan",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="actions",
            full_name="syft.core.plan.Plan.actions",
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
        ),
        _descriptor.FieldDescriptor(
            name="inputs",
            full_name="syft.core.plan.Plan.inputs",
            index=1,
            number=2,
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
        ),
        _descriptor.FieldDescriptor(
            name="outputs",
            full_name="syft.core.plan.Plan.outputs",
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
        ),
    ],
    extensions=[],
    nested_types=[
        _PLAN_INPUTSENTRY,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=125,
    serialized_end=356,
)

_PLAN_INPUTSENTRY.fields_by_name[
    "value"
].message_type = proto_dot_core_dot_pointer_dot_pointer__pb2._POINTER
_PLAN_INPUTSENTRY.containing_type = _PLAN
_PLAN.fields_by_name[
    "actions"
].message_type = proto_dot_core_dot_node_dot_common_dot_action_dot_action__pb2._ACTION
_PLAN.fields_by_name["inputs"].message_type = _PLAN_INPUTSENTRY
_PLAN.fields_by_name[
    "outputs"
].message_type = proto_dot_core_dot_pointer_dot_pointer__pb2._POINTER
DESCRIPTOR.message_types_by_name["Plan"] = _PLAN
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Plan = _reflection.GeneratedProtocolMessageType(
    "Plan",
    (_message.Message,),
    dict(
        InputsEntry=_reflection.GeneratedProtocolMessageType(
            "InputsEntry",
            (_message.Message,),
            dict(
                DESCRIPTOR=_PLAN_INPUTSENTRY,
                __module__="proto.core.plan.plan_pb2"
                # @@protoc_insertion_point(class_scope:syft.core.plan.Plan.InputsEntry)
            ),
        ),
        DESCRIPTOR=_PLAN,
        __module__="proto.core.plan.plan_pb2"
        # @@protoc_insertion_point(class_scope:syft.core.plan.Plan)
    ),
)
_sym_db.RegisterMessage(Plan)
_sym_db.RegisterMessage(Plan.InputsEntry)


_PLAN_INPUTSENTRY._options = None
# @@protoc_insertion_point(module_scope)
