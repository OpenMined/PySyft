# future
from __future__ import annotations

# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.bytes_pb2 import Bytes as Bytes_PB
from .primitive_interface import PyPrimitive


@serializable()
class Bytes(bytes, PyPrimitive):
    def __new__(cls, value: Any = None) -> Bytes:
        if value is None:
            value = b""

        return bytes.__new__(cls, value)

    def __init__(self, value: Any = None):
        if value is None:
            value = 0

        bytes.__init__(value)

    def upcast(self) -> bytes:
        return bytes(self)

    def _object2proto(self) -> Bytes_PB:
        bytes_pb = Bytes_PB()
        bytes_pb.data = self
        return bytes_pb

    @staticmethod
    def _proto2object(proto: Bytes_PB) -> Bytes:
        return Bytes(value=proto.data)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Bytes_PB
