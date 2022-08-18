# future
from __future__ import annotations

# stdlib
from typing import Any

# relative
from .primitive_interface import PyPrimitive


class Bytes(bytes, PyPrimitive):
    def __new__(cls, value: Any = None) -> Bytes:
        if value is None:
            value = b""

        return bytes.__new__(cls, value)

    def __init__(self, value: Any = None):
        if value is None:
            value = 0

        bytes.__init__(value)
        self.value = self

    def upcast(self) -> bytes:
        return bytes(self)
