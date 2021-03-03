# stdlib
from inspect import signature
from itertools import chain
from typing import Any
from typing import TypeVar

# syft absolute
from syft.lib.python import PyPrimitive
from syft.lib.python.primitive_factory import PrimitiveFactory
from syft.lib.python.types import SyPrimitiveRet

type_cache = dict()

T = TypeVar("T")


class IndexableTrait(type):
    """
    IndexableTrait is used as a metaclass for our UnionGenerator to be able to
    set __getitem__ on the class to enable the UnionGenerator[...] syntax.
    """

    def __getitem__(self, target_type) -> type:
        return Iterator(target_type)


class Iterator(type, metaclass=IndexableTrait):
    def __new__(cls, targeted_underlying_type: type) -> type:
        name = targeted_underlying_type.__name__ + "Iterator"

        if name in type_cache:
            return type_cache[name]

        bases = (_TemplateableIterator,)
        attrs, allowlist = generate_attrs_and_allowlist(
            _TemplateableIterator, targeted_underlying_type
        )
        new_type = super().__new__(cls, name, bases, attrs)
        globals()[name] = new_type
        type_cache[name] = allowlist
        return new_type


def generate_attrs_and_allowlist(templated_type, target_underlying_type):
    attrs = {}
    allowlist = {}

    def wrap_type_check(method):
        target_signature = signature(method)
        variable_checks = []

        for idx, (name, target_type) in enumerate(target_signature.parameters.items()):
            if target_type.annotation is T:
                variable_checks.append(idx)

        ret_type_check = target_signature.return_annotation is T

        def func(variable_checks):
            def _func(*args, **kwargs):
                arg_list = list(chain(args, kwargs.items()))

                for idx in variable_checks:
                    assert isinstance(arg_list[idx], target_underlying_type)

                result = method(*args, **kwargs)

                if ret_type_check:
                    assert isinstance(result, target_underlying_type)

                return result

            return _func

        return func(variable_checks)

    for method_path, type_path in iter_allowlist.items():
        method_name = method_path.rsplit(".", 1)[-1]
        method = getattr(templated_type, method_name)

        if not type_path:
            attrs[method_name] = wrap_type_check(method)
            allowlist[method_name] = target_underlying_type.__qualname__
        else:
            attrs[method_name] = getattr(_TemplateableIterator, method_name)
            allowlist[method_name] = type_path

    return attrs, allowlist


class _TemplateableIterator(PyPrimitive):
    def __init__(self, _ref: Any, max_len=None):
        super().__init__()
        self._obj_ref = _ref
        self._index = 0
        # self._id = UID()
        self.max_len = max_len
        self.exhausted = False

    def __iter__(self) -> "Iterator":
        return self

    def __reduce__(self) -> Any:
        # see these tests: test_valuesiterator_pickling and test_iterator_pickling
        raise TypeError(f"Pickling {type(self)} is not supported.")

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        if hasattr(other, "_obj_ref"):
            res = self._obj_ref == other._obj_ref
        else:
            res = self._obj_ref == other

        return PrimitiveFactory.generate_primitive(value=res)

    def __next__(self) -> T:
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
                setattr(self, "exhausted", True)
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
            raise e

    def dummy_example(self, type: T) -> T:
        return type


iter_allowlist = {
    "syft.lib.python.Iterator.__init__": "syft.lib.python.Iterator",
    "syft.lib.python.Iterator.__next__": None,
    "syft.lib.python.Iterator.__iter__": "syft.lib.python.Any",
    "syft.lib.python.Iterator.__eq__": "syft.lib.python.Bool",
    "syft.lib.python.Iterator.dummy_example": None,
}

int_iterable_type = Iterator[int]
iterable = int_iterable_type(([1, 2.5, 3]))

print(type_cache)
# {'intIterator': {'__init__': 'syft.lib.python.Iterator', '__next__': 'int', '__iter__': 'syft.lib.python.Any', '__eq__': 'syft.lib.python.Bool'}}

print(next(iterable))  # works

iterable.dummy_example(5)
iterable.dummy_example(5.5)
# print(next(iterable)) # will break
