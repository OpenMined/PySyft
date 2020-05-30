from syft.generic.abstract.hookable import chain_call
from syft.generic.abstract.hookable import hookable


def test_chain_call():
    class Chainable:
        def __init__(self, value):
            self.child = None
            self.value = value

        def chainable(self):
            return self.value

    c1 = Chainable(1)
    c1.child = Chainable(2)
    c1.child.child = Chainable(3)

    return_val = chain_call(c1, "chainable")

    assert return_val == [1, 2, 3]


def test_hookable():
    class Hookable:
        def __init__(self):
            self.child = None
            self.flags = {}

        def _before_set_flag(self, flag):
            self.flags[f"before_{flag}"] = True

        @hookable
        def set_flag(self, flag):
            self.flags[flag] = True

        def _after_set_flag(self, flag):
            self.flags[f"after_{flag}"] = True

    h = Hookable()

    h.set_flag("flag")

    assert h.flags["flag"] is True
    assert h.flags["before_flag"] is True
    assert h.flags["after_flag"] is True
