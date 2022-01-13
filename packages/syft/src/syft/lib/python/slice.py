"""
This source file aims to replace the standard slice object/function provided by Python
to be handled by the PySyft's abstract syntax tree data structure during a remote call.
"""

# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from ...core.common import UID
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
        id: Optional[UID] = None,
    ):
        """
        This class will receive start, stop, step and ID as valid parameters.

        :param start: Index/position where the slicing of the object starts.
        :param stop: Index/position which the slicing takes place. The slicing stops at index stop-1.
        :param step: Determines the increment between each index for slicing.
        :param id: PySyft's objects have an unique ID related to them.
        """

        # first, second, third
        if stop is None and step is None:
            # slice treats 1 arg as stop not start
            stop = start
            start = None

        self.value = slice(start, stop, step)
        self._id: UID = id if id else UID()

    @property
    def id(self) -> UID:
        """
        We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self == other.

        :param Any other: Object to be compared.
        :return: returns a PySyft boolean checking if self == other.
        :rtype: SyPrimitiveRet
        """
        res = self.value.__eq__(upcast(other))
        return PrimitiveFactory.generate_primitive(value=res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self >= other.

        :param Any other: Object to be compared.
        :return: returns a PySyft boolean checking if self >= other.
        :rtype: SyPrimitiveRet
        """
        res = self.value.__ge__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self > other.

        :param Any other: Object to be compared.
        :return: returns a PySyft boolean checking if self > other.
        :rtype: SyPrimitiveRet
        """
        res = self.value.__gt__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self =< other.

        :param Any other: Object to be compared.
        :return: returns a PySyft boolean checking if self =< other.
        :rtype: SyPrimitiveRet
        """
        res = self.value.__le__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self < other.

        :param Any other: Object to be compared.
        :return: returns a PySyft boolean checking if self < other.
        :rtype: SyPrimitiveRet
        """
        res = self.value.__lt__(upcast(other))  # type: ignore
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        """
        Compare if self != other.

        :param other: Object to be compared.
        :return: returns a PySyft boolean checking if self != other.
        :rtype: SyPrimitiveRet
        """
        res = self.value.__ne__(upcast(other))
        return PrimitiveFactory.generate_primitive(value=res)

    def __str__(self) -> str:
        """ Slice's string representation
            
            :return: String representation of this Slice object.
            :rtype: str
        """
        return self.value.__str__()

    def indices(self, index: int) -> tuple:
        """
        Assuming a sequence of length len, calculate the start and stop
        indices, and the stride length of the extended slice described by
        the Slice object. Out of bounds indices are clipped in
        a manner consistent with the handling of normal slices.
        
        :param int index: 
        :return: tuple of indices.
        :rtype: tuple
        """
        res = self.value.indices(index)
        return PrimitiveFactory.generate_primitive(value=res)

    @property
    def start(self) -> Optional[int]:
        """
        Index/position where the slicing of the object starts.
        
        :return: Index where the slicing starts.
        :rtype: int
        """
        return self.value.start

    @property
    def step(self) -> Optional[int]:
        """
        Increment between each index for slicing.
        
        :return: Slices's increment value.
        :rtype: int
        """
        return self.value.step

    @property
    def stop(self) -> Optional[int]:
        """
        Index/position which the slicing takes place.
        :return: Slice's stop value.
        :rtype: int
        """
        return self.value.stop

    def upcast(self) -> slice:
        """
        Returns the standard python slice object.
        
        :return: returns a default python slice object represented by this object instance.
        :rtype: slice
        """
        return self.value

    def _object2proto(self) -> Slice_PB:
        """
        Serialize  the Slice object instance returning a protobuf.
        
        :return: returns a protobuf object class representing this Slice object.
        :rtype: Slice_PB
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

        slice_pb.id.CopyFrom(sy.serialize(obj=self._id))

        return slice_pb

    @staticmethod
    def _proto2object(proto: Slice_PB) -> "Slice":
        """
        Deserialize a protobuf object creating a new Slice object instance.

        :param Slice_PB proto: Protobuf object representing a serialized slice object.
        :return: PySyft Slice object instance.
        :rtype: Slice
        """
        id_: UID = sy.deserialize(blob=proto.id)
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
            id=id_,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """
        Returns the proper Slice protobuf schema.
        
        :rtype: Slice_PB
        """
        return Slice_PB
