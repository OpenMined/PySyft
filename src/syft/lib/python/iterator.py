# stdlib
from typing import Any
from typing import Optional

# syft relative
from ...core.common.uid import UID
from ...decorators import syft_decorator
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet


class Iterator(PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, _ref: Any, max_len: Optional[int] = None):
        super().__init__()
        self._obj_ref = _ref
        self._index = 0
        self._id = UID()
        self.max_len = max_len
        self.exhausted = False

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iter__(self) -> "Iterator":
        return self

    def __reduce__(self) -> Any:
        # see these tests: test_valuesiterator_pickling and test_iterator_pickling
        raise TypeError(f"Pickling {type(self)} is not supported.")

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        if hasattr(other, "_obj_ref"):
            res = self._obj_ref == other._obj_ref
        else:
            res = self._obj_ref == other

        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
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
            if (max_len is not None and self._index >= max_len) or exhausted:
                setattr(self, "exhausted", True)
                raise StopIteration

            try:
                if hasattr(_obj_ref, "__next__"):
                    try:
                        obj = next(_obj_ref)
                    except Exception as e:
                        # print(f"{type(e)} on __next__ in {type(self)}. {e}")
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
                    setattr(self, "_obj_ref", _obj_ref.__iter__())
                    # obj = next(self._obj_ref) # just call self.__next__() instead
                    return self.__next__()
                else:
                    raise ValueError("Can't iterate through given object.")
            except StopIteration as e:
                setattr(self, "exhausted", True)
                raise e

            if hasattr(self, "_index"):
                self._index += 1
            return obj
        except Exception as e:
            # print(f"{type(e)} on __next__ in {type(self)}. {e}")
            raise e
