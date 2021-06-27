# syft absolute
import syft as sy


def test_ordereddict_reversed() -> None:
    s = sy.lib.python.collections.OrderedDict()
    s["1"] = 1
    s["2"] = 2
    s["3"] = 3

    assert s == {"1": 1, "2": 2, "3": 3}

    reverse_keys = ["3", "2", "1"]
    for i in reversed(s):
        print(i, reverse_keys)
        assert i == reverse_keys.pop(0)
