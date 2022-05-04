# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np

# relative
from ...lib.numpy.array import capnp_deserialize
from ...lib.numpy.array import capnp_serialize
from ..common.serde.capnp import CapnpModule
from ..common.serde.capnp import chunk_bytes
from ..common.serde.capnp import combine_bytes
from ..common.serde.capnp import get_capnp_schema
from ..common.serde.capnp import serde_magic_header
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import serializable
from ..common.serde.serialize import _serialize as serialize
from .config import DEFAULT_FLOAT_NUMPY_TYPE
from .config import DEFAULT_INT_NUMPY_TYPE
from .passthrough import PassthroughTensor  # type: ignore
from .passthrough import is_acceptable_simple_type  # type: ignore
from .smpc import context


@serializable(capnp_bytes=True)
class FixedPrecisionTensor(PassthroughTensor):

    # __attr_allowlist__ = ("child", "_base", "_precision", "_scale")

    def __init__(
        self,
        value: Union[int, float, np.ndarray] = None,
        base: int = 2,
        precision: int = 16,
    ) -> None:
        self._base = base
        self._precision = precision
        self._scale = base**precision
        if value is not None:
            # TODO :Should modify to be compatiable with torch.

            super().__init__(self.encode(value))
        else:
            super().__init__(None)

    def encode(self, value: Union[int, float, np.ndarray]) -> np.ndarray:
        encoded_value = np.array(self._scale * value, DEFAULT_INT_NUMPY_TYPE)
        return encoded_value

    @property
    def precision(self) -> int:
        """Get the precision for the FixedPrecisionTensor.

        Returns:
            int: precision.
        """
        return self._precision

    @property
    def base(self) -> int:
        """Get the base for the FixedPrecisionTensor.

        Returns:
            int: base
        """
        return self._base

    @property
    def scale(self) -> int:
        """Get the scale for the FixedPrecisionTensor.

        Returns:
            int: the scale.
        """
        return self._scale

    def decode(self) -> Any:
        # relative
        from .smpc.share_tensor import ShareTensor

        value = self.child.child if isinstance(self.child, ShareTensor) else self.child

        correction = (value < 0).astype(DEFAULT_INT_NUMPY_TYPE)
        dividend = value // self._scale - correction
        remainder = value % self._scale
        remainder += (
            (remainder == 0).astype(DEFAULT_INT_NUMPY_TYPE) * self._scale * correction
        )
        value = (
            dividend.astype(DEFAULT_FLOAT_NUMPY_TYPE)
            + remainder.astype(DEFAULT_FLOAT_NUMPY_TYPE) / self._scale
        )
        return value

    def sanity_check(
        self, other: Union[FixedPrecisionTensor, int, float, np.ndarray]
    ) -> FixedPrecisionTensor:
        if isinstance(other, FixedPrecisionTensor):
            if self.base != other.base or self.precision != other.precision:
                raise ValueError(
                    f"Base:{self.base,other.base} and Precision: "
                    + f"{self.precision, other.precision} should be same for "
                    + "computation on FixedPrecisionTensor"
                )
        elif is_acceptable_simple_type(other):
            other = FixedPrecisionTensor(
                value=other, base=self.base, precision=self.precision
            )
        else:
            raise ValueError(f"Invalid type for FixedPrecisionTensor: {type(other)}")

        return other

    def __add__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            res.child = self.child + other
        else:
            other = self.sanity_check(other)
            res.child = self.child + other.child
        return res

    def __sub__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            res.child = self.child - other
        else:
            other = self.sanity_check(other)
            res.child = self.child - other.child
        return res

    def __mul__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            res.child = self.child * other
        else:
            other = self.sanity_check(other)
            context.FPT_CONTEXT["seed_id_locations"] = context.SMPC_CONTEXT.get(
                "seed_id_locations", None
            )
            res.child = self.child * other.child
            res = res / self.scale

        return res

    def __matmul__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, np.ndarray) and other.dtype == np.dtype("bool"):
            raise ValueError("Should not get  a boolan array to matmul")
        else:
            other = self.sanity_check(other)
            context.FPT_CONTEXT["seed_id_locations"] = context.SMPC_CONTEXT.get(
                "seed_id_locations", None
            )
            res.child = self.child @ other.child
            res = res / self.scale

        return res

    def __truediv__(
        self, other: Union[int, np.integer, FixedPrecisionTensor]
    ) -> FixedPrecisionTensor:
        if isinstance(other, FixedPrecisionTensor):
            raise ValueError("We do not support Private Division yet.")

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child / other
        if isinstance(res.child, np.ndarray):
            res.child = res.child.astype(DEFAULT_INT_NUMPY_TYPE)
        return res

    def transpose(
        self, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child.transpose(*args, **kwargs)
        return res

    @property
    def T(self) -> FixedPrecisionTensor:
        return self.transpose()

    # TODO: Remove after moving private compare to sharetensor level
    def __lt__(self, other: Any) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        if isinstance(other, FixedPrecisionTensor):
            res.child = (self.child < other.child) * 1
        else:
            res.child = (self.child < other) * 1
        return res

    def __gt__(self, other: Any) -> FixedPrecisionTensor:

        if isinstance(other, FixedPrecisionTensor):
            value = (self.child > other.child) * 1
        else:
            value = (self.child > other) * 1

        res = FixedPrecisionTensor(
            value=value, base=self._base, precision=self._precision
        )
        return res

    def concatenate(
        self, other: FixedPrecisionTensor, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> FixedPrecisionTensor:
        if not isinstance(other, FixedPrecisionTensor):
            raise NotImplementedError

        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child.concatenate(other.child, *args, **kwargs)

        return res

    def all(self) -> bool:
        return self.child.all()

    def copy(self, order: Optional[str] = "K") -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child.copy(order=order)
        return res

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> FixedPrecisionTensor:
        res = FixedPrecisionTensor(base=self._base, precision=self._precision)
        res.child = self.child.sum(axis=axis)
        return res

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="fixed_precision_tensor.capnp")

        fpt_struct: CapnpModule = schema.FPT  # type: ignore
        fpt_msg = fpt_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        fpt_msg.magicHeader = serde_magic_header(type(self))

        # child of FPT tensor could either be ShareTensor or np.ndarray
        if isinstance(self.child, np.ndarray):
            chunk_bytes(capnp_serialize(self.child, to_bytes=True), "child", fpt_msg)
            fpt_msg.isNumpy = True
        else:
            chunk_bytes(serialize(self.child, to_bytes=True), "child", fpt_msg)  # type: ignore
            fpt_msg.isNumpy = False

        fpt_msg.base = self.base
        fpt_msg.precision = self.precision

        # to pack or not to pack?
        # to_bytes = fpt_msg.to_bytes()

        return fpt_msg.to_bytes_packed()

    @staticmethod
    def _bytes2object(buf: bytes) -> FixedPrecisionTensor:
        schema = get_capnp_schema(schema_file="fixed_precision_tensor.capnp")
        fpt_struct: CapnpModule = schema.FPT  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # to pack or not to pack?
        # fpt_msg = fpt_struct.from_bytes(buf, traversal_limit_in_words=2 ** 64 - 1)
        fpt_msg = fpt_struct.from_bytes_packed(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )

        if fpt_msg.isNumpy:
            child = capnp_deserialize(combine_bytes(fpt_msg.child), from_bytes=True)
        else:
            child = deserialize(combine_bytes(fpt_msg.child), from_bytes=True)

        base = fpt_msg.base
        precision = fpt_msg.precision

        res = FixedPrecisionTensor(base=base, precision=precision)
        res.child = child
        return res
