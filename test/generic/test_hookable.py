from syft.generic.abstract.hookable import map_chain_call
from syft.generic.abstract.hookable import reduce_chain_call
from syft.generic.abstract.hookable import hookable


def test_reduce_chain_call():
    class Reduceable:
        def __init__(self, value):
            self.child = None
            self.value = value

        def reduceable(self, value):
            return value + self.value

    c1 = Reduceable(1)
    c1.child = Reduceable(2)
    c1.child.child = Reduceable(3)

    return_val = reduce_chain_call(c1, "reduceable", 0)

    assert return_val == 6


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

    return_val = map_chain_call(c1, "mappable")

    assert return_val == [1, 2, 3]


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


def test_hooks_propagate_return_val():
    class Alterable:
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

    a = Alterable({"a": 1}, {"after_a": 1})
    a.child = Alterable({"b": 2}, {"after_b": 2})
    a.child.child = Alterable({"c": 3}, {"after_c": 3})

    result = a.get_value()

    assert result == {"a": 1, "after_a": 1, "after_b": 2, "after_c": 3}
