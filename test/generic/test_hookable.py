from syft.generic.abstract.hookable import map_chain_call
from syft.generic.abstract.hookable import reduce_chain_call
from syft.generic.abstract.hookable import hookable


def test_map_chain_call():
    class Mappable:
        def __init__(self, value):
            self.child = None
            self.value = value

        def mappable(self):
            return self.value

    c1 = Mappable(1)
    c1.child = Mappable(2)
    c1.child.child = Mappable(3)

    return_val, *_ = map_chain_call(c1, "mappable")

    assert return_val == [1, 2, 3]


def test_map_chain_call_modifies_args():
    class MappableMod:
        def __init__(self, value):
            self.child = None
            self.value = value

        def mappable(self, *args, **kwargs):
            arg_value = args[0]
            kwarg_key, kwarg_value = list(kwargs.items())[0]
            result = self.value + arg_value + kwarg_value
            return (result, [arg_value + 1], {kwarg_key: kwarg_value + 1})

    c1 = MappableMod(1)
    c1.child = MappableMod(1)
    c1.child.child = MappableMod(1)

    return_val, *_ = map_chain_call(c1, "mappable", 0, test=0)

    assert return_val == [1, 3, 5]  # [ (1+0+0), (1+1+1), (1+2+2) ]


def test_reduce_chain_call():
    class Reduceable:
        def __init__(self, value):
            self.child = None
            self.value = value

        def reduceable(self, accumulator):
            return accumulator + self.value

    c1 = Reduceable(1)
    c1.child = Reduceable(2)
    c1.child.child = Reduceable(3)

    return_val, *_ = reduce_chain_call(c1, "reduceable", 0)

    assert return_val == 6  # 1 + 2 + 3


def test_reduce_chain_call_modifies_args():
    class ReduceableMod:
        def __init__(self, value):
            self.child = None
            self.value = value

        def reduceable(self, accumulator, *args, **kwargs):
            arg_value = args[0]
            kwarg_key, kwarg_value = list(kwargs.items())[0]
            result = accumulator + self.value + arg_value + kwarg_value
            return (result, [arg_value + 1], {kwarg_key: kwarg_value + 1})

    c1 = ReduceableMod(1)
    c1.child = ReduceableMod(1)
    c1.child.child = ReduceableMod(1)

    return_val, *_ = reduce_chain_call(c1, "reduceable", 0, 0, test=0)

    assert return_val == 9  # (1+0+0) + (1+1+1) + (1+2+2)


def test_hooks_get_called():
    class Hookable:
        def __init__(self):
            self.child = None
            self.flags = {}

        def _before_set_flag(self, flag):
            self.flags[f"before_{flag}"] = True

        @hookable
        def set_flag(self, flag):
            self.flags[flag] = True

        def _after_set_flag(self, obj, flag):
            self.flags[f"after_{flag}"] = True

    h = Hookable()

    h.set_flag("flag")

    assert h.flags["flag"] is True
    assert h.flags["before_flag"] is True
    assert h.flags["after_flag"] is True


def test_hooks_can_modify_args():
    class ArgMod:
        def __init__(self, value):
            self.child = None
            self.value = value

        def _before_concat(self, other):
            reversed = other[::-1]
            return (None, [reversed], {})

        @hookable
        def concat(self, other):
            return self.value + other

    a = ArgMod("first")
    a.child = ArgMod("second")
    a.child.child = ArgMod("third")

    result = a.concat("string")

    assert result == "firstgnirts"


def test_hooks_can_modify_return_val():
    class ReturnMod:
        def __init__(self, value, after):
            self.child = None
            self.value = value
            self.after = after

        @hookable
        def get_value(self):
            return self.value

        def _after_get_value(self, return_val):
            # Merge dictionaries
            return_val = {**return_val, **self.after}
            return return_val

    r = ReturnMod({"a": 1}, {"after_a": 1})
    r.child = ReturnMod({"b": 2}, {"after_b": 2})
    r.child.child = ReturnMod({"c": 3}, {"after_c": 3})

    result = r.get_value()

    assert result == {"a": 1, "after_a": 1, "after_b": 2, "after_c": 3}
