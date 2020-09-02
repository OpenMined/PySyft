# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...decorators import syft_decorator
from ...proto.lib.python.int_pb2 import Int as Int_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive


class MetaSyInt(type):

    shadow_type = int

    # --------------------- A op B = C ---------------------
    operand_overrides = {
        "__add__": "+",
        "__and__": "and",
        "__div__": "/",
        "__floordiv__": "//",
        "__ge__": ">=",
        "__gt__": ">",
        "__le__": "<=",
        "__lshift__": "<<",
        "__lt__": "<",
        "__mod__": "%",
        "__mul__": "*",
        "__or__": "or",
        "__pow__": "^",
        "__radd__": "+",
        "__rand__": "and",
        "__rdiv__": "/",
        "__rlshift__": "<<",
        "__rmod__": "%",
        "__rmul__": "*",
        "__ror__": "or",
        "__rpow__": "^",
        "__rrshift__": ">>",
        "__rshift__": ">>",
        "__rsub__": "-",
        "__rxor__": "xor",
        "__sub__": "-",
        "__truediv__": "/",
        "__xor__": "xor",
    }

    def __eq__(self, other: Any) -> bool:
        """Checks to see if two PyPrimitives are equal ignoring their ids.

        This checks to see whether this PyPrimitives is equal to another by
        comparing whether they have the same .data objects. These objects
        come with their own __eq__ function which we assume to be correct.

        :param other: this is the other PyPrimitives to be compared with
        :type other: Any (note this must be Any or __eq__ fails on other types)
        :return: returns True/False based on whether the objects are the same
        :rtype: bool
        """

        try:
            if isinstance(other, SyInt) or issubclass(type(other), SyInt):
                return int(self._value) == int(other._value)  # type: ignore
            return int(self._value) == other  # type: ignore
        except Exception:
            return False

    @staticmethod
    def make_operand_method(op: str, symbol: str) -> Callable:
        def run_operand(a: "SyInt", b: Any) -> Union["SyInt", bool]:
            # print("running hijacked operand", type(a), type(b), op)
            try:
                method = getattr(a._value, op, None)
                if method is not None:
                    if isinstance(b, SyInt) or issubclass(type(b), SyInt):
                        result = method(b._value)
                    else:
                        result = method(b)
                else:
                    raise Exception(f"No {op} found on {type(a)}")

                # these methods just want a boolean comparison result
                if op in ["__gt__", "__ge__", "__lt__", "__le__"]:
                    return bool(result)
                else:
                    # return result as a new instance of type a with a.id
                    syPrimitiveType = type(a)
                    newValue = syPrimitiveType(result)
                    newValue._id = a.id
                    return newValue
            except Exception as e:
                raise TypeError(
                    "unsupported operand type(s) for: "
                    + f"'{a.__repr__()}' {symbol} '{b.__repr__()}'. "
                    + f"{e}"
                )

        return run_operand

    def debug_method(method: Callable, op: str) -> Callable:
        method_type = type(method).__name__
        # print("binding method", op, method_type)

        def run_method(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            # print(
            #     f"running super {op} with args: ",
            #     [type(a) for a in args],
            #     [type(v) for k, v in kwargs.items()],
            # )
            if method_type in ["getset_descriptor"]:
                # its a property so no args or () call
                return method
            else:
                return method(*args, **kwargs)

        return run_method

    def __new__(
        cls: Type, name: str, bases: Tuple[Type, ...], dct: Dict[str, Any]
    ) -> "MetaSyInt":
        sy_meta_cls = super().__new__(cls, name, (MetaSyInt.shadow_type,), dct)

        overwritten = []
        for method, symbol in MetaSyInt.operand_overrides.items():
            overwritten.append(method)  # keep track

            # override the method with our wrapper method
            setattr(
                sy_meta_cls,
                method,
                MetaSyInt.make_operand_method(op=method, symbol=symbol),
            )

        setattr(sy_meta_cls, "__eq__", MetaSyInt.__eq__)

        # --------------------- useful debug code ---------------------
        # lets see what isnt being overwritten
        # add debug to non overwritten methods
        for k in dir(sy_meta_cls):
            if k in ["__class__", "__dict__", "__init__"]:
                # skip these dunder methods
                continue
            if k not in overwritten:
                # print(f"{MetaSyInt.shadow_type}.{k} has not been overwritten")
                super_method = getattr(sy_meta_cls, k)
                if type(super_method).__name__ in ["property"]:
                    # skip properties
                    # print(f"Skipping {super_method}")
                    continue
                setattr(
                    sy_meta_cls, k, MetaSyInt.debug_method(method=super_method, op=k)
                )
        return sy_meta_cls  # type: ignore


class SyInt(PyPrimitive, metaclass=MetaSyInt):
    _value: int
    _id: UID

    def __init__(self, value: Any = 0, base: Any = 10) -> None:
        if isinstance(value, str):
            value = int(value, base)
        self._value = value
        self._id: UID = UID()

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    @property
    def icon(self) -> str:
        return "🗿"

    @property
    def pprint(self) -> str:
        type_name = f"{self.class_name} {type(self._value).__name__}"
        output = f"{self.icon} ({type_name}) {self._value}"
        return output

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Int_PB:
        int_pb = Int_PB()
        int_pb.data = self
        int_pb.id.CopyFrom(serialize(self.id))
        return int_pb

    @staticmethod
    def _proto2object(proto: Int_PB) -> "Int":
        int_id: UID = deserialize(blob=proto.id)
        return Int(value=proto.data, id=int_id)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Int_PB


class Int(int, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __new__(
        cls, value: Any = None, base: Any = 10, id: Optional[UID] = None
    ) -> "Int":
        if value is None:
            value = 0

        if isinstance(value, str):
            return int.__new__(cls, value, base)  # type: ignore

        return int.__new__(cls, value)  # type: ignore

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Any = None, base: Any = 10, uid: Optional[UID] = None):
        if value is None:
            value = 0

        int.__init__(value)

        self._id: UID = UID() if uid is None else uid

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> PyPrimitive:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __radd__(self, other: Any) -> PyPrimitive:
        res = super().__radd__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: Any) -> PyPrimitive:
        res = super().__sub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rsub__(self, other: Any) -> PyPrimitive:
        res = super().__rsub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmul__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__rmul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __floordiv__(self, other: Any) -> PyPrimitive:
        res = super().__floordiv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__truediv__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mod__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__mod__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rmod__(self, other: Any) -> PyPrimitive:
        res = super().__rmod__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __pow__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__pow__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rpow__(self, other: Any) -> PyPrimitive:
        res = super().__rpow__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lshift__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__lshift__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rlshift__(self, other: Any) -> PyPrimitive:
        res = super().__rlshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rshift__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__rshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rrshift__(self, other: Any) -> PyPrimitive:
        res = super().__rrshift__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __and__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__and__(other)
        return PrimitiveFactory.generate_primitive(res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rand__(self, other: Any) -> PyPrimitive:
        res = super().__rand__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __xor__(self, other: Any) -> PyPrimitive:
        res = super(Int, self).__xor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __rxor__(self, other: Any) -> PyPrimitive:
        res = super().__rxor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __or__(self, other: Any) -> PyPrimitive:
        res = super().__or__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ror__(self, other: Any) -> Any:
        res = super().__ror__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> PyPrimitive:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> PyPrimitive:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> PyPrimitive:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> PyPrimitive:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Int_PB:
        int_pb = Int_PB()
        int_pb.data = self
        int_pb.id.CopyFrom(serialize(self.id))
        return int_pb

    @staticmethod
    def _proto2object(proto: Int_PB) -> "Int":
        int_id: UID = deserialize(blob=proto.id)
        return Int(value=proto.data, id=int_id)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Int_PB
