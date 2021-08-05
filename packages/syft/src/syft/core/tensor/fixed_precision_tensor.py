# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor

# relative
from ...core.common.serde.serializable import Serializable
from ...proto.core.tensor.fixed_precision_tensor_pb2 import (
    FixedPrecisionTensor as FixedPrecisionTensor_PB,
)
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import bind_protobuf
from ..common.serde.serialize import _serialize as serialize


@bind_protobuf
class FixedPrecisionTensor(PassthroughTensor, Serializable):
    def __init__(
        self, value: Optional[Any] = None, base: int = 10, precision: int = 3
    ) -> None:
        self._base = base
        self._precision = precision
        self._scale = base ** precision
        if value is not None:
            fpt_value = self._scale * value
            encoded_value = fpt_value.astype(np.int64)
            super().__init__(encoded_value)

    def decode(self) -> Any:
        correction = (self.child < 0).astype(np.int64)
        dividend = self.child // self._scale - correction
        remainder = self.child % self._scale
        remainder += (remainder == 0).astype(np.int64) * self._scale * correction
        value = dividend.astype(np.float32) + remainder.astype(np.float32) / self._scale
        return value

    def __add__(self, other: Any) -> "FixedPrecisionTensor":
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child + other.child
        return res

    def __sub__(self, other: Any) -> "FixedPrecisionTensor":
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child - other.child
        return res

    def _object2proto(self) -> "FixedPrecisionTensor_PB":
        # syft absolute
        from syft.core.tensor.share_tensor import ShareTensor
        from syft.core.tensor.tensor import Tensor

        if isinstance(self.child, Tensor):
            return FixedPrecisionTensor_PB(
                tensor=serialize(self.child), base=self._base, precision=self._precision
            )
        elif isinstance(self.child, ShareTensor):
            return FixedPrecisionTensor_PB(
                share=serialize(self.child), base=self._base, precision=self._precision
            )

        return FixedPrecisionTensor_PB(
            array=serialize(self.child), base=self._base, precision=self._precision
        )

    @staticmethod
    def _proto2object(proto: FixedPrecisionTensor_PB) -> "FixedPrecisionTensor":
        res = FixedPrecisionTensor(base=proto.base, precision=proto.precision)

        # Put it manually since we send it already encoded
        if proto.HasField("tensor"):
            res.child = deserialize(proto.tensor)
        elif proto.HasField("share"):
            res.child = deserialize(proto.share)
        else:
            res.child = deserialize(proto.array)
        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return FixedPrecisionTensor_PB
