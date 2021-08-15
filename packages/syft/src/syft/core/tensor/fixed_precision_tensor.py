# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
import torch

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor

# relative
from ... import logger
from ...core.common.serde.serializable import Serializable
from ...proto.core.tensor.fixed_precision_tensor_pb2 import (
    FixedPrecisionTensor as FixedPrecisionTensor_PB,
)
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import bind_protobuf
from ..common.serde.serialize import _serialize as serialize
from .smpc.utils import is_int_array
from .smpc.utils import is_int_tensor
from .util import implements


@bind_protobuf
class FixedPrecisionTensor(PassthroughTensor, Serializable):
    def __init__(
        self, value: Optional[Any] = None, base: int = 10, precision: int = 3
    ) -> None:
        self._base = base
        self._precision = precision
        self._scale = base ** precision
        if value is not None:
            # syft absolute
            from syft.core.tensor.tensor import Tensor

            if not isinstance(value, Tensor):
                value = Tensor(child=value)

            fpt_value = self._scale * value
            encoded_value = fpt_value.astype(np.int64)
            super().__init__(encoded_value)

    def copy_tensor(self) -> "FixedPrecisionTensor":
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child
        return res

    def decode(self) -> Any:
        correction = (self.child < 0).astype(np.int64)
        dividend = self.child // self._scale - correction
        remainder = self.child % self._scale
        remainder = (
            remainder + (remainder == 0).astype(np.int64) * self._scale * correction
        )
        value = dividend.astype(np.float32) + remainder.astype(np.float32) / self._scale
        return value

    def add(self, y: Any) -> "FixedPrecisionTensor":
        import pdb; pdb.set_trace()
        if not isinstance(y, FixedPrecisionTensor):
            y = FixedPrecisionTensor(
                value=y, base=self._base, precision=self._precision
            )

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        # Already encoded, don't pass it in the constructor such that we do not re-encode
        res.child = self.child + y.child
        return res

    def sub(self, y: Any) -> "FixedPrecisionTensor":
        if not isinstance(y, FixedPrecisionTensor):
            y = FixedPrecisionTensor(
                value=y, base=self._base, precision=self._precision
            )
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child - y.child
        return res

    def rsub(self, y: Any) -> "FixedPrecisionTensor":
        if not isinstance(y, FixedPrecisionTensor):
            y = FixedPrecisionTensor(base=self._base, precision=self._precision)

        if self._base != y._base:
            raise ValueError(f"Different base for operators {self._base} and {y._base}")
        if self._precision != y._precision:
            # TODO: Maybe take the highest precision? (need to think a little about this)
            logger.warning(
                f"Different precision for operators {self._precision} and {y._precision}"
            )

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = y.child - self.child
        return res

    def mul(
        self, y: Union[int, float, torch.Tensor, np.ndarray]
    ) -> "FixedPrecisionTensor":
        if isinstance(y, int) or is_int_tensor(y) or is_int_array(y):
            res = FixedPrecisionTensor(base=self._base, precision=self._precision)
            res.child = self.child * y
            return res
        else:
            raise ValueError("Multiplication works only with integer values")

    def truediv(self, y: int) -> "FixedPrecisionTensor":
        if not isinstance(y, int):
            raise ValueError(
                f"Truediv should have as divisor an integer,, but found {type(y)}"
            )

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        # Manually place it such that we do not re-encode it
        res.child = value = self.child // y
        return res

    def mod(self, y: int) -> "FixedPrecisionTensor":
        if not isinstance(y, int):
            raise ValueError(
                f"Modulo should have as divisor an integer, but found {type(y)}"
            )

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        # Manually place it such that we do not re-encode it
        res.child = self.child % y
        return res

    def _object2proto(self) -> FixedPrecisionTensor_PB:
        # syft absolute
        from syft.core.tensor.smpc.share_tensor import ShareTensor
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

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __mul__ = mul
    __rmul__ = mul
    __truediv__ = truediv
    __div__ = truediv
    __mod__ = mod
