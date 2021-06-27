# syft absolute
import syft as sy


def test_string_reversed() -> None:
    s = sy.lib.python.String("dog")
    t = ""
    for i in reversed(s):
        t += i
    assert t == "god"
