# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from .. import python as py
from ...core.common.serde.serializable import serializable
from ...core.common.uid import UID
from ...logger import traceback_and_raise
from ...proto.lib.python.iterator_pb2 import Iterator as Iterator_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet


@serializable()
class Iterator(PyPrimitive):
    def __init__(self, _ref: Any, max_len: Optional[int] = None):
        super().__init__()
        self._obj_ref = _ref
        self._index = 0
        self._id = UID()
        self.max_len = max_len
        self.exhausted = False

    def __iter__(self) -> "Iterator":
        return self

    def __len__(self) -> int:
        try:
            return len(self._obj_ref)
        except Exception as e:
            traceback_and_raise(e)

    def __reduce__(self) -> Any:
        # see these tests: test_valuesiterator_pickling and test_iterator_pickling
        raise TypeError(f"Pickling {type(self)} is not supported.")

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        if hasattr(other, "_obj_ref"):
            res = self._obj_ref == other._obj_ref
        else:
            res = self._obj_ref == other

        return PrimitiveFactory.generate_primitive(value=res)

    def __next__(self) -> Any:
        # we need to do lots of getattr / setattr because some times the __next__
        # method gets called with a generator
        try:
            if hasattr(self, "_obj_ref"):
                _obj_ref = self._obj_ref
            else:
                # we got handed a generator directly into __next__
                # happens in test_reversed_iterator
                _obj_ref = self

            # max_len == None means the _ref could update while iterating. While that
            # shouldn't happen with a IteratorPointer, it can happen on a local Iterator.
            # If thats the case we just calculate it each time. Importantly we need to
            # still set exhausted otherwise the test case in list_test.py wont pass.
            max_len = None
            if hasattr(self, "max_len"):
                max_len = self.max_len

            if max_len is None:
                try:
                    if hasattr(_obj_ref, "__len__"):
                        max_len = _obj_ref.__len__()
                except AttributeError:
                    # I am not sure why this happens on some types
                    pass

            exhausted = getattr(self, "exhausted", False)
            self_index = getattr(self, "_index", 0)
            if (max_len is not None and self_index >= max_len) or exhausted:
                self.exhausted = True
                raise StopIteration

            try:
                if hasattr(_obj_ref, "__next__"):
                    try:
                        obj = next(_obj_ref)
                    except Exception as e:
                        if type(e) is StopIteration:
                            raise e
                        if type(e) is AttributeError:
                            # no _mapping exhausted?
                            raise StopIteration()
                        if type(e) is NameError:
                            # free after use?
                            raise StopIteration()

                        # test_dictitems_contains_use_after_free wants us to StopIteration
                        # test_merge_and_mutate and test_mutating_iteration wants us to
                        # raise a RuntimeError
                        # see:
                        # def test_dictitems_contains_use_after_free(self):
                        # Lets RuntimeError for now
                        raise RuntimeError

                elif hasattr(_obj_ref, "__getitem__") and hasattr(self, "_index"):
                    obj = _obj_ref[self._index]
                elif hasattr(_obj_ref, "__iter__"):
                    # collections.abc.* KeysView, ValuesView, ItemsView end up here
                    # they do not have __next__ or __getitem__ but they do have __iter__
                    # so we can just replace our self._obj_ref and keep going
                    self._obj_ref = _obj_ref.__iter__()
                    # obj = next(self._obj_ref) # just call self.__next__() instead
                    return self.__next__()
                else:
                    raise ValueError("Can't iterate through given object.")
            except StopIteration as e:
                self.exhausted = True
                raise e

            if hasattr(self, "_index"):
                self._index += 1
            return obj
        except Exception as e:
            raise e

    def upcast(self) -> Any:
        return iter(self._obj_ref)

    # TODO: Fix based on message from Tudor Cebere
    # So, when we add a new builtin type we want to have feature parity with cython ones.
    # When we tried to do this for iterators in the early days we had some problems when the iterators are infinite
    # (most likely an iterator from a generator). This pattern is common in functional programming, when you use
    # infinite iterators for different purposes. I then said that it makes sense to force the user to exhaust the
    # iterator himself and then to serde the type. Here, it might be a bit problematic because somebody might slip
    # in this kind of iterator and when we exhaust it (through list conversion), we go into infinite computation.
    # And there are similar edge cases to this.

    def _object2proto(self) -> Iterator_PB:
        id_ = sy.serialize(obj=self._id)
        obj_ref_ = sy.serialize(py.list.List(list(self._obj_ref)), to_bytes=True)
        index_ = self._index
        max_len_ = self.max_len
        exhausted_ = self.exhausted
        return Iterator_PB(
            id=id_,
            obj_ref=obj_ref_,
            index=index_,
            max_len=max_len_,
            exhausted=exhausted_,
        )

    @staticmethod
    def _proto2object(proto: Iterator_PB) -> "Iterator":
        id_: UID = sy.deserialize(blob=proto.id)
        obj_ref_ = sy.deserialize(blob=proto.obj_ref, from_bytes=True)
        index_ = proto.index
        max_len_ = proto.max_len
        exhausted_ = proto.exhausted

        new_iter = Iterator(_ref=obj_ref_, max_len=max_len_)

        new_iter._index = index_
        new_iter.exhausted = exhausted_
        new_iter._id = id_

        return new_iter

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Iterator_PB
