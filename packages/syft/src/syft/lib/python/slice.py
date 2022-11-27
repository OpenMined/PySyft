"""
This source file aims to replace the standard slice object/function provided by Python
to be handled by the PySyft's abstract syntax tree data structure during a remote call.
"""

# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.slice_pb2 import Slice as Slice_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet
from .util import upcast


@serializable()  # This decorator turns this class serializable.
class Slice(PyPrimitive):
    def __init__(
        self,
        start: Any = None,
        stop: Optional[Any] = None,
        step: Optional[Any] = None,
    ):
        """
        This class will receive start, stop, step  valid parameters.

        Args:
            start (Any): Index/position where the slicing of the object starts.
            stop (Any): Index/position which the slicing takes place. The slicing stops at index stop-1.
            step (Any): Determines the increment between each index for slicing.
        """

        # first, second, third
        if stop is None and step is None:
            # slice treats 1 arg as stop not start
            stop = start
            start = None

        self.value = slice(start, stop, step)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self == other.

        Args:
            other (Any): Object to be compared.
        Returns:
            SyPrimitiveRet: returns a PySyft boolean format checking if self == other.
        """
        res = self.value.__eq__(upcast(other))
        return PrimitiveFactory.generate_primitive(value=res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self >= other.

        Args:
            other (Any): Object to be compared.
        Returns:
            SyPrimitiveRet: returns a PySyft boolean format checking if self >= other.
        """
        res = self.value.__ge__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self > other.

        Args:
            other (Any): Object to be compared.
        Returns:
            SyPrimitiveRet: returns a PySyft boolean format checking if self > other.
        """
        res = self.value.__gt__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self <= other.

        Args:
            other (Any): Object to be compared.
        Returns:
            SyPrimitiveRet: returns a PySyft boolean format checking if self <= other.
        """
        res = self.value.__le__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self < other.

        Args:
            other (Any): Object to be compared.
        Returns:
            SyPrimitiveRet: returns a PySyft boolean format checking if self < other.
        """
        res = self.value.__lt__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self != other.

        Args:
            other (Any): Object to be compared.
        Returns:
            SyPrimitiveRet: returns a PySyft boolean format checking if self != other.
        """
        res = self.value.__ne__(upcast(other))
        return PrimitiveFactory.generate_primitive(value=res)

    def __str__(self) -> str:
        """Slice's string representation

        Returns:
            str: The string representation of this Slice object.
        """
        return self.value.__str__()

    def indices(self, index: int) -> tuple:
        """
        Assuming a sequence of length len, calculate the start and stop
        indices, and the stride length of the extended slice described by
        the Slice object. Out of bounds indices are clipped in
        a manner consistent with the handling of normal slices.

        Args:
            index (int): Input index.
        Returns:
            Tuple: A tuple of concrete indices for a range of length len.
        """
        res = self.value.indices(index)
        return PrimitiveFactory.generate_primitive(value=res)

    @property
    def start(self) -> Optional[int]:
        """
        Index/position where the slicing of the object starts.

        Returns:
            int: Index where the slicing starts.
        """
        return self.value.start

    @property
    def step(self) -> Optional[int]:
        """
        Increment between each index for slicing.

        Returns:
            int: Slices' increment value.
        """
        return self.value.step

    @property
    def stop(self) -> Optional[int]:
        """
        Index/position which the slicing takes place.

        Returns:
            int: Slices' stop value.
        """
        return self.value.stop

    def upcast(self) -> slice:
        """
        Returns the standard python slice object.

        Returns:
            slice: returns a default python slice object represented by this object instance.
        """
        return self.value

    def _object2proto(self) -> Slice_PB:
        """
        Serialize  the Slice object instance returning a protobuf.

        Returns:
            Slice_PB: returns a protobuf object class representing this Slice object.
        """
        slice_pb = Slice_PB()
        if self.start:
            slice_pb.start = self.start
            slice_pb.has_start = True

        if self.stop:
            slice_pb.stop = self.stop
            slice_pb.has_stop = True

        if self.step:
            slice_pb.step = self.step
            slice_pb.has_step = True

        return slice_pb

    @staticmethod
    def _proto2object(proto: Slice_PB) -> "Slice":
        """
        Deserialize a protobuf object creating a new Slice object instance.

        Args:
            proto (Slice_PB): Protobuf object representing a serialized slice object.
        Returns:
            Slice: PySyft Slice object instance.
        """
        start = None
        stop = None
        step = None
        if proto.has_start:
            start = proto.start

        if proto.has_stop:
            stop = proto.stop

        if proto.has_step:
            step = proto.step

        return Slice(
            start=start,
            stop=stop,
            step=step,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """
        Returns the proper Slice protobuf schema.

        Returns:
            Slice_PB: Returns the Slice's Protobuf class definition.
        """
        return Slice_PB
