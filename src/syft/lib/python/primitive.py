# from typing import Any
# from typing import Optional
# from typing import List
# from typing import Union
#
# from google.protobuf.reflection import GeneratedProtocolMessageType
#
# from ...core.common.serde import _deserialize
# from ...core.common.serde import Serializable
# from ...decorators.syft_decorator_impl import syft_decorator
#
# from ...core.common.uid import UID
#
# from syft.core.store.storeable_object import StorableObject
# from ...util import aggressive_set_attr
#
#
# @syft_decorator(typechecking=True)
# def isprimitive(value: Any) -> bool:
#     # check primitive types first
#     if type(value) in [type(None), bool, int, float, str]:
#         return True
#
#     return False
#
#
# class PyPrimitive(Serializable):
#     data: Union[None, bool, int, float, str]
#     _id: UID
#
#     @syft_decorator(typechecking=True)
#     def __init__(
#         self, data: Union[None, bool, int, float, str], id: Optional[UID] = None
#     ):
#         """This initializer allows the creation of python primitive types that have an
#         associated ID and can be serialized.
#
#         :param data: one of the supported python primitive types to be wrapped
#         :type data: Union[None, bool, int, float, str]
#         :param id: an override which can be used to set an ID for this object
#             manually. This is probably only used for deserialization.
#         :type id: UID
#         :return: returns a PyPrimitive
#         :rtype: PyPrimitive
#
#         """
#         self.data = data
#
#         if id is None:
#             id = UID()
#
#         self._id: UID = id
#
#         # while this class is never used as a simple wrapper,
#         # it's possible that sub-classes of this class will be.
#         super().__init__()
#
#     @property
#     def icon(self) -> str:
#         return "🗿"
#
#     @property
#     def pprint(self) -> str:
#         output = (
#             f"{self.icon} ({self.class_name} {type(self.data).__name__}) {self.data}"
#         )
#         return output
#
#     @property
#     def class_name(self) -> str:
#         return str(self.__class__.__name__)
#
#     @property
#     def id(self) -> UID:
#         """We reveal PyPrimitive.id as a property to discourage users and
#         developers of Syft from modifying .id attributes after an object
#         has been initialized.
#
#         :return: returns the unique id of the object
#         :rtype: UID
#         """
#         return self._id
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __eq__(self, other: Any) -> bool:
#         """Checks to see if two PyPrimitives are equal ignoring their ids.
#
#         This checks to see whether this PyPrimitives is equal to another by
#         comparing whether they have the same .data objects. These objects
#         come with their own __eq__ function which we assume to be correct.
#
#         :param other: this is the other PyPrimitives to be compared with
#         :type other: Any (note this must be Any or __eq__ fails on other types)
#         :return: returns True/False based on whether the objects are the same
#         :rtype: bool
#         """
#
#         try:
#             if isinstance(other, PyPrimitive):
#                 return self.data == other.data
#             return self.data == other
#         except Exception:
#             return False
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __len__(self) -> int:
#         return len(self.data)  # type: ignore
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __bool__(self) -> bool:
#         if type(self.data) is not str and hasattr(self.data, "__bool__"):
#             return self.data.__bool__()  # type: ignore
#         else:
#             return False
#
#     # --------------------- A op B = C ---------------------
#     @staticmethod
#     def run_operand(
#         a: "PyPrimitive", b: Any, op: str, symbol: str
#     ) -> Union["PyPrimitive", bool]:
#         try:
#             method = getattr(a.data, op, None)
#             if method is not None:
#                 if isinstance(b, PyPrimitive):
#                     print("got method and pyprimivites", method, a.data, b.data)
#                     result = method(b.data)
#                 else:
#                     result = method(b)
#             else:
#                 print("here???")
#                 raise Exception("No method")
#             if op in ["__gt__", "__ge__", "__lt__", "__le__"]:
#                 return bool(result)
#             else:
#                 return PyPrimitive(data=result, id=a.id)
#         except Exception:
#             raise TypeError(
#                 "unsupported operand type(s) for {symbol}: "
#                 + f"'{a.__repr__()}' and '{b.__repr__()}'"
#             )
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __add__(self, other: Any) -> "PyPrimitive":
#         return PyPrimitive.run_operand(a=self, b=other, op="__add__", symbol="+")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __radd__(self, other: Any) -> "PyPrimitive":
#         return self.__add__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __sub__(self, other: Any) -> "PyPrimitive":
#         return PyPrimitive.run_operand(a=self, b=other, op="__sub__", symbol="-")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rsub__(self, other: Any) -> "PyPrimitive":
#         return self.__sub__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __mul__(self, other: Any) -> "PyPrimitive":
#         return PyPrimitive.run_operand(a=self, b=other, op="__mul__", symbol="*")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rmul__(self, other: Any) -> Any:
#         return self.__mul__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __div__(self, other: Any) -> Any:
#         print(self, other)
#         return PyPrimitive.run_operand(a=self, b=other, op="__div__", symbol="/")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rdiv__(self, other: Any) -> Any:
#         return self.__div__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __floordiv__(self, other: Any) -> Any:
#         print(self, other)
#         return PyPrimitive.run_operand(a=self, b=other, op="__floordiv__", symbol="//")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __truediv__(self, other: Any) -> Any:
#         print(self, other)
#         return PyPrimitive.run_operand(a=self, b=other, op="__truediv__", symbol="/")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __mod__(self, other: Any) -> Any:
#         return PyPrimitive.run_operand(a=self, b=other, op="__mod__", symbol="%")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rmod__(self, other: Any) -> Any:
#         return self.__mod__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __pow__(self, other: Any) -> Any:
#         return PyPrimitive.run_operand(a=self, b=other, op="__pow__", symbol="**")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rpow__(self, other: Any) -> Any:
#         return self.__pow__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __lshift__(self, other: Any) -> Any:
#         return PyPrimitive.run_operand(a=self, b=other, op="__lshift__", symbol="<<")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rlshift__(self, other: Any) -> Any:
#         return self.__lshift__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rshift__(self, other: Any) -> Any:
#         return PyPrimitive.run_operand(a=self, b=other, op="__rshift__", symbol=">>")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rrshift__(self, other: Any) -> Any:
#         return self.__rshift__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __and__(self, other: Any) -> Any:
#         return PyPrimitive.run_operand(a=self, b=other, op="__and__", symbol="and")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rand__(self, other: Any) -> Any:
#         return self.__and__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __xor__(self, other: Any) -> Any:
#         return PyPrimitive.run_operand(a=self, b=other, op="__xor__", symbol="xor")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __rxor__(self, other: Any) -> Any:
#         return self.__xor__(other)
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __or__(self, other: Any) -> Any:
#         return PyPrimitive.run_operand(a=self, b=other, op="__or__", symbol="or")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __ror__(self, other: Any) -> Any:
#         return self.__or__(other)
#
#     # --------------------- A op B = True or False ---------------------
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __ge__(self, other: Any) -> bool:
#         return PyPrimitive.run_operand(a=self, b=other, op="__ge__", symbol=">")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __lt__(self, other: Any) -> bool:
#         return PyPrimitive.run_operand(a=self, b=other, op="__lt__", symbol="<")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __le__(self, other: Any) -> bool:
#         return PyPrimitive.run_operand(a=self, b=other, op="__le__", symbol="<=")
#
#     @syft_decorator(typechecking=True, prohibit_args=False)
#     def __gt__(self, other: Any) -> bool:
#         return PyPrimitive.run_operand(a=self, b=other, op="__gt__", symbol=">=")
#
#     @syft_decorator(typechecking=True)
#     def __repr__(self) -> str:
#         """Returns a human-readable version of the PyPrimitive
#
#         Return a human-readable representation of the PyPrimitive with brackets
#         so that it can be easily spotted when nested inside of the human-
#         readable representations of other objects."""
#
#         return f"<{type(self).__name__}:{self.id.value} {self.data}>"
#
#     @syft_decorator(typechecking=True)
#     def repr_short(self) -> str:
#         """Returns a SHORT human-readable version of SpecificLocation
#
#         Return a SHORT human-readable version of the ID which
#         makes it print nicer when embedded (often alongside other
#         UID objects) within other object __repr__ methods."""
#
#         return f"<{type(self).__name__}:{self.id.repr_short()}>"
#
#     @syft_decorator(typechecking=True)
#     def _object2proto(self) -> PyPrimitive_PB:
#         """Returns a protobuf serialization of self.
#
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms the current object into the corresponding
#         Protobuf object so that it can be further serialized.
#
#         :return: returns a protobuf object
#         :rtype: PyPrimitive_PB
#
#         .. note::
#             This method is purely an internal method. Please use object.serialize() or one of
#             the other public serialization methods if you wish to serialize an
#             object.
#         """
#
#         proto = PyPrimitive_PB()
#         proto.id.CopyFrom(self.id.serialize())
#         t = type(self.data)
#
#         if isinstance(self.data, type(None)):
#             proto.type = PyPrimitive_PB.NONE
#         elif t == bool:
#             proto.type = PyPrimitive_PB.BOOL
#             proto.int = int(self.data)  # type: ignore
#         elif t == int:
#             proto.type = PyPrimitive_PB.INT
#             proto.int = self.data
#         elif t == float:
#             proto.type = PyPrimitive_PB.FLOAT
#             proto.float = self.data
#         elif t == str:
#             proto.type = PyPrimitive_PB.STRING
#             proto.str = self.data
#         else:
#             raise Exception(f"Cant serialize {self.__repr__()} with {type(self.data)}")
#
#         return proto
#
#     @staticmethod
#     def _proto2object(proto: PyPrimitive_PB) -> "PyPrimitive":
#         """Creates a PyPrimitive from a protobuf
#
#         As a requirement of all objects which inherit from Serializable,
#         this method transforms a protobuf object into an instance of this class.
#
#         :return: returns an instance of PyPrimitive
#         :rtype: PyPrimitive
#
#         .. note::
#             This method is purely an internal method. Please use syft.deserialize()
#             if you wish to deserialize an object.
#         """
#
#         t = proto.type
#         data: Union[None, bool, int, float, str]
#
#         if t == PyPrimitive_PB.NONE:
#             data = None
#         elif t == PyPrimitive_PB.BOOL:
#             data = bool(proto.int)
#         elif t == PyPrimitive_PB.INT:
#             data = proto.int
#         elif t == PyPrimitive_PB.FLOAT:
#             data = proto.float
#         elif t == PyPrimitive_PB.STRING:
#             data = proto.str
#         else:
#             raise Exception(f"Cant deserialize {proto.id} {proto.type} with {proto}")
#
#         return PyPrimitive(id=_deserialize(blob=proto.id), data=data)
#
#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         """ Return the type of protobuf object which stores a class of this type
#
#         As a part of serialization and deserialization, we need the ability to
#         lookup the protobuf object type directly from the object type. This
#         static method allows us to do this.
#
#         Importantly, this method is also used to create the reverse lookup ability within
#         the metaclass of Serializable. In the metaclass, it calls this method and then
#         it takes whatever type is returned from this method and adds an attribute to it
#         with the type of this class attached to it. See the MetaSerializable class for details.
#
#         :return: the type of protobuf object which corresponds to this class.
#         :rtype: GeneratedProtocolMessageType
#
#         """
#
#         return PyPrimitive_PB
#
#
# class PyPrimitiveWrapper(StorableObject):
#     def __init__(self, value: object):
#         super().__init__(
#             data=value,
#             id=getattr(value, "id", UID()),
#             tags=getattr(value, "tags", []),
#             description=getattr(value, "description", ""),
#         )
#         self.value = value
#
#     def _data_object2proto(self) -> PyPrimitive_PB:
#         return self.data._object2proto()
#
#     @staticmethod
#     def _data_proto2object(proto: PyPrimitive_PB) -> PyPrimitive:
#         return PyPrimitive._proto2object(proto)
#
#     @staticmethod
#     def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
#         return PyPrimitive_PB
#
#     @staticmethod
#     def get_wrapped_type() -> type:
#         return PyPrimitive
#
#     @staticmethod
#     def construct_new_object(
#         id: UID, data: StorableObject, tags: List[str], description: Optional[str]
#     ) -> object:
#         data._id = id
#         data.tags = tags
#         data.description = description
#         return data
#
#
# aggressive_set_attr(
#     obj=PyPrimitive, name="serializable_wrapper_type", attr=PyPrimitiveWrapper
# )
