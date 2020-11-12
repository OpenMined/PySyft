"""
File copied from cpython test suite:
https://github.com/python/cpython/blob/3.8/Lib/test/string_tests.py

Common tests shared by test_unicode, test_userstring and test_bytes.
"""

# stdlib
import string
import struct
import sys
from test import support

# third party
import pytest

# syft absolute
from syft.lib import python


class Sequence:
    def __init__(self, seq="wxyz"):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return self.seq[i]


class BadSeq1(Sequence):
    def __init__(self):
        self.seq = [7, "hello", 123]

    def __str__(self):
        return "{0} {1} {2}".format(*self.seq)


class BadSeq2(Sequence):
    def __init__(self):
        self.seq = ["a", "b", "c"]

    def __len__(self):
        return 8


# These tests are for buffers of values (bytes) and not
# specific to character interpretation, used for bytes objects
# and various string implementations

# The type to be tested
# Change in subclasses to change the behaviour of fixtesttype()
type2test = None

# Whether the "contained items" of the container are integers in
# range(0, 256) (i.e. bytes, bytearray) or strings of length 1
# (str)
contains_bytes = False


# check that obj.method(*args) returns result
def checkequal(result, obj, methodname, *args, **kwargs):
    realresult = getattr(obj, methodname)(*args, **kwargs)
    assert result == realresult
    # if the original is returned make sure that
    # this doesn't happen with subclasses


# check that obj.method(*args) raises exc
def checkraises(exc, obj, methodname, *args):
    with pytest.raises(exc) as e_info:
        getattr(obj, methodname)(*args)
    assert str(e_info) != ""


# call obj.method(*args) without any checks
def checkcall(obj, methodname, *args):
    getattr(obj, methodname)(*args)


@pytest.mark.slow
def test_find():
    checkequal(0, python.String("abcdefghiabc"), "find", python.String("abc"))
    checkequal(9, python.String("abcdefghiabc"), "find", python.String("abc"), 1)
    checkequal(-1, python.String("abcdefghiabc"), "find", python.String("def"), 4)

    checkequal(0, python.String("abc"), "find", python.String(""), 0)
    checkequal(3, python.String("abc"), "find", python.String(""), 3)
    checkequal(-1, python.String("abc"), "find", python.String(""), 4)

    # to check the ability to pass None as defaults
    checkequal(2, python.String("rrarrrrrrrrra"), "find", python.String("a"))
    checkequal(12, python.String("rrarrrrrrrrra"), "find", python.String("a"), 4)
    checkequal(-1, python.String("rrarrrrrrrrra"), "find", python.String("a"), 4, 6)
    checkequal(12, python.String("rrarrrrrrrrra"), "find", python.String("a"), 4, None)
    checkequal(2, python.String("rrarrrrrrrrra"), "find", python.String("a"), None, 6)

    checkraises(TypeError, python.String("hello"), "find")

    if contains_bytes:
        checkequal(-1, python.String("hello"), "find", 42)
    else:
        checkraises(TypeError, python.String("hello"), "find", 42)

    checkequal(0, python.String(""), "find", python.String(""))
    checkequal(-1, python.String(""), "find", python.String(""), 1, 1)
    checkequal(-1, python.String(""), "find", python.String(""), sys.maxsize, 0)

    checkequal(-1, python.String(""), "find", python.String("xx"))
    checkequal(-1, python.String(""), "find", python.String("xx"), 1, 1)
    checkequal(-1, python.String(""), "find", python.String("xx"), sys.maxsize, 0)

    # issue 7458
    checkequal(
        -1, python.String("ab"), "find", python.String("xxx"), sys.maxsize + 1, 0
    )

    # For a variety of combinations,
    #    verify that str.find() matches __contains__
    #    and that the found substring is really at that location
    charset = [python.String(""), python.String("a"), python.String("b")]
    digits = 4
    base = len(charset)
    teststrings = set()
    for i in range(base ** digits):
        entry = []
        for j in range(digits):
            i, m = divmod(i, base)
            entry.append(charset[m])
        teststrings.add(python.String("").join(entry))
    for i in teststrings:
        for j in teststrings:
            loc = i.find(j)
            r1 = loc != -1
            r2 = j in i
            assert r1 == r2
            if loc != -1:
                idx = loc + len(j)
                assert i[loc:idx] == j


@pytest.mark.slow
def test_rfind():
    checkequal(9, python.String("abcdefghiabc"), "rfind", python.String("abc"))
    checkequal(12, python.String("abcdefghiabc"), "rfind", "")
    checkequal(0, python.String("abcdefghiabc"), "rfind", python.String("abcd"))
    checkequal(-1, python.String("abcdefghiabc"), "rfind", python.String("abcz"))

    checkequal(3, python.String("abc"), "rfind", python.String(""), 0)
    checkequal(3, python.String("abc"), "rfind", python.String(""), 3)
    checkequal(-1, python.String("abc"), "rfind", python.String(""), 4)

    # to check the ability to pass None as defaults
    checkequal(12, python.String("rrarrrrrrrrra"), "rfind", python.String("a"))
    checkequal(12, python.String("rrarrrrrrrrra"), "rfind", python.String("a"), 4)
    checkequal(-1, python.String("rrarrrrrrrrra"), "rfind", python.String("a"), 4, 6)
    checkequal(12, python.String("rrarrrrrrrrra"), "rfind", python.String("a"), 4, None)
    checkequal(2, python.String("rrarrrrrrrrra"), "rfind", python.String("a"), None, 6)

    checkraises(TypeError, python.String("hello"), "rfind")

    if contains_bytes:
        checkequal(-1, python.String("hello"), "rfind", 42)
    else:
        checkraises(TypeError, python.String("hello"), "rfind", 42)

    # For a variety of combinations,
    #    verify that str.rfind() matches __contains__
    #    and that the found substring is really at that location
    charset = [python.String(""), python.String("a"), python.String("b")]
    digits = 3
    base = len(charset)
    teststrings = set()
    for i in range(base ** digits):
        entry = []
        for j in range(digits):
            i, m = divmod(i, base)
            entry.append(charset[m])
        teststrings.add(python.String("").join(entry))
    for i in teststrings:
        for j in teststrings:
            loc = i.rfind(j)
            r1 = loc != -1
            r2 = j in i
            assert r1 == r2
            if loc != -1:
                assert i[loc : loc + len(j)] == j  # noqa: E203

    # issue 7458
    checkequal(
        -1, python.String("ab"), "rfind", python.String("xxx"), sys.maxsize + 1, 0
    )

    # issue #15534
    checkequal(0, python.String("<......\u043c..."), "rfind", "<")


def test_index():
    checkequal(0, python.String("abcdefghiabc"), "index", python.String(""))
    checkequal(3, python.String("abcdefghiabc"), "index", python.String("def"))
    checkequal(0, python.String("abcdefghiabc"), "index", python.String("abc"))
    checkequal(9, python.String("abcdefghiabc"), "index", "abc", 1)

    checkraises(
        ValueError, python.String("abcdefghiabc"), "index", python.String("hib")
    )
    checkraises(
        ValueError, python.String("abcdefghiab"), "index", python.String("abc"), 1
    )
    checkraises(
        ValueError, python.String("abcdefghi"), "index", python.String("ghi"), 8
    )
    checkraises(
        ValueError, python.String("abcdefghi"), "index", python.String("ghi"), -1
    )

    # to check the ability to pass None as defaults
    checkequal(2, python.String("rrarrrrrrrrra"), "index", python.String("a"))
    checkequal(12, python.String("rrarrrrrrrrra"), "index", python.String("a"), 4)
    checkraises(
        ValueError, python.String("rrarrrrrrrrra"), "index", python.String("a"), 4, 6
    )
    checkequal(12, python.String("rrarrrrrrrrra"), "index", python.String("a"), 4, None)
    checkequal(2, python.String("rrarrrrrrrrra"), "index", python.String("a"), None, 6)

    checkraises(TypeError, python.String("hello"), "index")

    if contains_bytes:
        checkraises(ValueError, python.String("hello"), "index", 42)
    else:
        checkraises(TypeError, python.String("hello"), "index", 42)


def test_rindex():
    checkequal(12, python.String("abcdefghiabc"), "rindex", python.String(""))
    checkequal(3, python.String("abcdefghiabc"), "rindex", python.String("def"))
    checkequal(9, python.String("abcdefghiabc"), "rindex", python.String("abc"))
    checkequal(0, python.String("abcdefghiabc"), "rindex", python.String("abc"), 0, -1)

    checkraises(
        ValueError, python.String("abcdefghiabc"), "rindex", python.String("hib")
    )
    checkraises(
        ValueError, python.String("defghiabc"), "rindex", python.String("def"), 1
    )
    checkraises(ValueError, python.String("defghiabc"), "rindex", "abc", 0, -1)
    checkraises(
        ValueError, python.String("abcdefghi"), "rindex", python.String("ghi"), 0, 8
    )
    checkraises(
        ValueError, python.String("abcdefghi"), "rindex", python.String("ghi"), 0, -1
    )

    # to check the ability to pass None as defaults
    checkequal(12, python.String("rrarrrrrrrrra"), "rindex", "a")
    checkequal(12, python.String("rrarrrrrrrrra"), "rindex", "a", 4)
    checkraises(
        ValueError, python.String("rrarrrrrrrrra"), "rindex", python.String("a"), 4, 6
    )
    checkequal(
        12, python.String("rrarrrrrrrrra"), "rindex", python.String("a"), 4, None
    )
    checkequal(2, python.String("rrarrrrrrrrra"), "rindex", python.String("a"), None, 6)

    checkraises(TypeError, "hello", "rindex")

    if contains_bytes:
        checkraises(ValueError, "hello", "rindex", 42)
    else:
        checkraises(TypeError, python.String("hello"), "rindex", 42)


def test_lower():
    checkequal(
        python.String("hello"),
        python.String("HeLLo"),
        "lower",
    )
    checkequal(python.String("hello"), python.String("hello"), "lower")
    checkraises(TypeError, python.String("hello"), "lower", 42)


def test_upper():
    checkequal(python.String("HELLO"), python.String("HeLLo"), "upper")
    checkequal(python.String("HELLO"), python.String("HELLO"), "upper")
    checkraises(TypeError, "hello", "upper", 42)


def test_expandtabs():
    checkequal(
        python.String("abc\rab      def\ng       hi"),
        python.String("abc\rab\tdef\ng\thi"),
        "expandtabs",
    )
    checkequal(
        python.String("abc\rab      def\ng       hi"),
        python.String("abc\rab\tdef\ng\thi"),
        "expandtabs",
        8,
    )
    checkequal(
        "abc\rab  def\ng   hi", python.String("abc\rab\tdef\ng\thi"), "expandtabs", 4
    )
    checkequal(
        python.String("abc\r\nab      def\ng       hi"),
        python.String("abc\r\nab\tdef\ng\thi"),
        "expandtabs",
    )
    checkequal(
        python.String("abc\r\nab      def\ng       hi"),
        "abc\r\nab\tdef\ng\thi",
        "expandtabs",
        8,
    )
    checkequal(
        python.String("abc\r\nab  def\ng   hi"),
        "abc\r\nab\tdef\ng\thi",
        "expandtabs",
        4,
    )
    checkequal(
        python.String("abc\r\nab\r\ndef\ng\r\nhi"),
        python.String("abc\r\nab\r\ndef\ng\r\nhi"),
        "expandtabs",
        4,
    )
    # check keyword args
    checkequal(
        python.String("abc\rab      def\ng       hi"),
        python.String("abc\rab\tdef\ng\thi"),
        "expandtabs",
        tabsize=8,
    )
    checkequal(
        "abc\rab  def\ng   hi",
        python.String("abc\rab\tdef\ng\thi"),
        "expandtabs",
        tabsize=4,
    )

    checkequal("  a\n b", python.String(" \ta\n\tb"), "expandtabs", 1)

    checkraises(TypeError, python.String("hello"), "expandtabs", 42, 42)
    # This test is only valid when sizeof(int) == sizeof(void*) == 4.
    if sys.maxsize < (1 << 32) and struct.calcsize("P") == 4:
        checkraises(OverflowError, python.String("\ta\n\tb"), "expandtabs", sys.maxsize)


def test_split():
    # by a char
    checkequal(
        [python.String("a"), python.String("b"), "c", python.String("d")],
        python.String("a|b|c|d"),
        "split",
        "|",
    )
    checkequal(["a|b|c|d"], python.String("a|b|c|d"), "split", "|", 0)
    checkequal(
        ["a", python.String("b|c|d")],
        python.String("a|b|c|d"),
        "split",
        python.String("|"),
        1,
    )
    checkequal(["a", "b", "c|d"], python.String("a|b|c|d"), "split", "|", 2)
    checkequal(
        ["a", "b", "c", "d"], python.String("a|b|c|d"), "split", python.String("|"), 3
    )
    checkequal(
        ["a", python.String("b"), python.String("c"), python.String("d")],
        python.String("a|b|c|d"),
        "split",
        "|",
        4,
    )
    checkequal(
        ["a", "b", "c", "d"],
        python.String("a|b|c|d"),
        "split",
        python.String("|"),
        sys.maxsize - 2,
    )
    checkequal(["a|b|c|d"], python.String("a|b|c|d"), "split", python.String("|"), 0)
    checkequal(
        [python.String("a"), python.String(""), python.String("b||c||d")],
        python.String("a||b||c||d"),
        "split",
        python.String("|"),
        2,
    )
    checkequal([python.String("abcd")], python.String("abcd"), "split", "|")
    checkequal([""], python.String(""), "split", "|")
    checkequal(
        [python.String("endcase "), python.String("")],
        python.String("endcase |"),
        "split",
        python.String("|"),
    )
    checkequal(
        ["", python.String(" startcase")],
        python.String("| startcase"),
        "split",
        python.String("|"),
    )
    checkequal(
        [python.String(""), python.String("bothcase"), python.String("")],
        python.String("|bothcase|"),
        "split",
        "|",
    )
    checkequal(
        ["a", "", python.String("b\x00c\x00d")],
        python.String("a\x00\x00b\x00c\x00d"),
        "split",
        "\x00",
        2,
    )

    checkequal(["a"] * 20, (python.String("a|") * 20)[:-1], "split", python.String("|"))
    checkequal(
        [python.String("a")] * 15 + [python.String("a|a|a|a|a")],
        ("a|" * 20)[:-1],
        "split",
        "|",
        15,
    )

    # by string
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a//b//c//d"),
        "split",
        python.String("//"),
    )
    checkequal(
        [python.String("a"), python.String("b//c//d")],
        python.String("a//b//c//d"),
        "split",
        "//",
        1,
    )
    checkequal(
        ["a", "b", "c//d"], python.String("a//b//c//d"), "split", python.String("//"), 2
    )
    checkequal(["a", "b", "c", "d"], python.String("a//b//c//d"), "split", "//", 3)
    checkequal(
        ["a", "b", "c", python.String("d")],
        python.String("a//b//c//d"),
        "split",
        "//",
        4,
    )
    checkequal(
        ["a", "b", "c", "d"],
        python.String("a//b//c//d"),
        "split",
        python.String("//"),
        sys.maxsize - 10,
    )
    checkequal(
        [python.String("a//b//c//d")],
        python.String("a//b//c//d"),
        "split",
        python.String("//"),
        0,
    )
    checkequal(
        [python.String("a"), python.String(""), "b////c////d"],
        python.String("a////b////c////d"),
        "split",
        "//",
        2,
    )
    checkequal(
        [python.String(""), python.String(" bothcase "), python.String("")],
        python.String("test bothcase test"),
        "split",
        python.String("test"),
    )
    checkequal(
        [python.String("a"), python.String("bc")],
        python.String("abbbc"),
        "split",
        python.String("bb"),
    )
    checkequal(
        [python.String(""), python.String("")],
        python.String("aaa"),
        "split",
        python.String("aaa"),
    )
    checkequal(
        [python.String("aaa")], python.String("aaa"), "split", python.String("aaa"), 0
    )
    checkequal(
        [python.String("ab"), python.String("ab")],
        python.String("abbaab"),
        "split",
        python.String("ba"),
    )
    checkequal(
        [python.String("aaaa")], python.String("aaaa"), "split", python.String("aab")
    )
    checkequal([python.String("")], python.String(""), "split", python.String("aaa"))
    checkequal(
        [python.String("aa")], python.String("aa"), "split", python.String("aaa")
    )
    checkequal(
        [python.String("A"), python.String("bobb")],
        python.String("Abbobbbobb"),
        "split",
        python.String("bbobb"),
    )
    checkequal(
        [python.String("A"), python.String("B"), python.String("")],
        python.String("AbbobbBbbobb"),
        "split",
        python.String("bbobb"),
    )

    checkequal(
        [python.String("a")] * 20,
        (python.String("aBLAH") * 20)[:-4],
        "split",
        python.String("BLAH"),
    )
    checkequal(
        [python.String("a")] * 20,
        (python.String("aBLAH") * 20)[:-4],
        "split",
        python.String("BLAH"),
        19,
    )
    checkequal(
        [python.String("a")] * 18 + [python.String("aBLAHa")],
        (python.String("aBLAH") * 20)[:-4],
        "split",
        python.String("BLAH"),
        18,
    )

    # with keyword args
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a|b|c|d"),
        "split",
        sep=python.String("|"),
    )
    checkequal(
        [python.String("a"), python.String("b|c|d")],
        python.String("a|b|c|d"),
        "split",
        python.String("|"),
        maxsplit=1,
    )
    checkequal(
        [python.String("a"), python.String("b|c|d")],
        python.String("a|b|c|d"),
        "split",
        sep=python.String("|"),
        maxsplit=1,
    )
    checkequal(
        [python.String("a"), python.String("b|c|d")],
        python.String("a|b|c|d"),
        "split",
        maxsplit=1,
        sep=python.String("|"),
    )
    checkequal(
        [python.String("a"), python.String("b c d")],
        python.String("a b c d"),
        "split",
        maxsplit=1,
    )

    # argument type
    checkraises(TypeError, python.String("hello"), "split", 42, 42, 42)

    # null case
    checkraises(ValueError, python.String("hello"), "split", python.String(""))
    checkraises(ValueError, python.String("hello"), "split", python.String(""), 0)


def test_rsplit():
    # by a char
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
    )
    checkequal(
        [python.String("a|b|c"), python.String("d")],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
        1,
    )
    checkequal(
        [python.String("a|b"), python.String("c"), python.String("d")],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
        2,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
        3,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
        4,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
        sys.maxsize - 100,
    )
    checkequal(
        [python.String("a|b|c|d")],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
        0,
    )
    checkequal(
        [python.String("a||b||c"), python.String(""), python.String("d")],
        python.String("a||b||c||d"),
        "rsplit",
        python.String("|"),
        2,
    )
    checkequal(
        [python.String("abcd")], python.String("abcd"), "rsplit", python.String("|")
    )
    checkequal([python.String("")], python.String(""), "rsplit", python.String("|"))
    checkequal(
        [python.String(""), python.String(" begincase")],
        python.String("| begincase"),
        "rsplit",
        python.String("|"),
    )
    checkequal(
        [python.String("endcase "), python.String("")],
        python.String("endcase |"),
        "rsplit",
        python.String("|"),
    )
    checkequal(
        [python.String(""), python.String("bothcase"), python.String("")],
        python.String("|bothcase|"),
        "rsplit",
        python.String("|"),
    )

    checkequal(
        [python.String("a\x00\x00b"), python.String("c"), python.String("d")],
        python.String("a\x00\x00b\x00c\x00d"),
        "rsplit",
        python.String("\x00"),
        2,
    )

    checkequal(
        [python.String("a")] * 20,
        (python.String("a|") * 20)[:-1],
        "rsplit",
        python.String("|"),
    )
    checkequal(
        [python.String("a|a|a|a|a")] + [python.String("a")] * 15,
        (python.String("a|") * 20)[:-1],
        "rsplit",
        python.String("|"),
        15,
    )

    # by string
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a//b//c//d"),
        "rsplit",
        python.String("//"),
    )
    checkequal(
        [python.String("a//b//c"), python.String("d")],
        python.String("a//b//c//d"),
        "rsplit",
        python.String("//"),
        1,
    )
    checkequal(
        [python.String("a//b"), python.String("c"), python.String("d")],
        python.String("a//b//c//d"),
        "rsplit",
        python.String("//"),
        2,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a//b//c//d"),
        "rsplit",
        python.String("//"),
        3,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a//b//c//d"),
        "rsplit",
        python.String("//"),
        4,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a//b//c//d"),
        "rsplit",
        python.String("//"),
        sys.maxsize - 5,
    )
    checkequal(
        [python.String("a//b//c//d")],
        python.String("a//b//c//d"),
        "rsplit",
        python.String("//"),
        0,
    )
    checkequal(
        [python.String("a////b////c"), python.String(""), python.String("d")],
        python.String("a////b////c////d"),
        "rsplit",
        python.String("//"),
        2,
    )
    checkequal(
        [python.String(""), python.String(" begincase")],
        python.String("test begincase"),
        "rsplit",
        python.String("test"),
    )
    checkequal(
        [python.String("endcase "), python.String("")],
        python.String("endcase test"),
        "rsplit",
        python.String("test"),
    )
    checkequal(
        [python.String(""), python.String(" bothcase "), python.String("")],
        python.String("test bothcase test"),
        "rsplit",
        python.String("test"),
    )
    checkequal(
        [python.String("ab"), python.String("c")],
        python.String("abbbc"),
        "rsplit",
        python.String("bb"),
    )
    checkequal(
        [python.String(""), python.String("")],
        python.String("aaa"),
        "rsplit",
        python.String("aaa"),
    )
    checkequal(
        [python.String("aaa")], python.String("aaa"), "rsplit", python.String("aaa"), 0
    )
    checkequal(
        [python.String("ab"), python.String("ab")],
        python.String("abbaab"),
        "rsplit",
        python.String("ba"),
    )
    checkequal(
        [python.String("aaaa")], python.String("aaaa"), "rsplit", python.String("aab")
    )
    checkequal([python.String("")], python.String(""), "rsplit", python.String("aaa"))
    checkequal(
        [python.String("aa")], python.String("aa"), "rsplit", python.String("aaa")
    )
    checkequal(
        [python.String("bbob"), python.String("A")],
        python.String("bbobbbobbA"),
        "rsplit",
        python.String("bbobb"),
    )
    checkequal(
        [python.String(""), python.String("B"), python.String("A")],
        python.String("bbobbBbbobbA"),
        "rsplit",
        python.String("bbobb"),
    )

    checkequal(
        [python.String("a")] * 20,
        (python.String("aBLAH") * 20)[:-4],
        "rsplit",
        python.String("BLAH"),
    )
    checkequal(
        [python.String("a")] * 20,
        (python.String("aBLAH") * 20)[:-4],
        "rsplit",
        python.String("BLAH"),
        19,
    )
    checkequal(
        [python.String("aBLAHa")] + [python.String("a")] * 18,
        (python.String("aBLAH") * 20)[:-4],
        "rsplit",
        python.String("BLAH"),
        18,
    )

    # with keyword args
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a|b|c|d"),
        "rsplit",
        sep=python.String("|"),
    )
    checkequal(
        [python.String("a|b|c"), python.String("d")],
        python.String("a|b|c|d"),
        "rsplit",
        python.String("|"),
        maxsplit=1,
    )
    checkequal(
        [python.String("a|b|c"), python.String("d")],
        python.String("a|b|c|d"),
        "rsplit",
        sep=python.String("|"),
        maxsplit=1,
    )
    checkequal(
        [python.String("a|b|c"), python.String("d")],
        python.String("a|b|c|d"),
        "rsplit",
        maxsplit=1,
        sep=python.String("|"),
    )
    checkequal(
        [python.String("a b c"), python.String("d")],
        python.String("a b c d"),
        "rsplit",
        maxsplit=1,
    )

    # argument type
    checkraises(TypeError, python.String("hello"), "rsplit", 42, 42, 42)

    # null case
    checkraises(ValueError, python.String("hello"), "rsplit", python.String(""))
    checkraises(ValueError, python.String("hello"), "rsplit", python.String(""), 0)


def test_replace():
    EQ = checkequal

    # Operations on the empty string
    EQ(
        python.String(""),
        python.String(""),
        "replace",
        python.String(""),
        python.String(""),
    )
    EQ(
        python.String("A"),
        python.String(""),
        "replace",
        python.String(""),
        python.String("A"),
    )
    EQ(
        python.String(""),
        python.String(""),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String(""),
        python.String(""),
        "replace",
        python.String("A"),
        python.String("A"),
    )
    EQ(
        python.String(""),
        python.String(""),
        "replace",
        python.String(""),
        python.String(""),
        100,
    )
    EQ(
        python.String(""),
        python.String(""),
        "replace",
        python.String(""),
        python.String(""),
        sys.maxsize,
    )

    # interleave (from==python.String(""), 'to' gets inserted everywhere)
    EQ(
        python.String("A"),
        python.String("A"),
        "replace",
        python.String(""),
        python.String(""),
    )
    EQ(
        python.String("*A*"),
        python.String("A"),
        "replace",
        python.String(""),
        python.String("*"),
    )
    EQ(
        python.String("*1A*1"),
        python.String("A"),
        "replace",
        python.String(""),
        python.String("*1"),
    )
    EQ(
        python.String("*-#A*-#"),
        python.String("A"),
        "replace",
        python.String(""),
        python.String("*-#"),
    )
    EQ(
        python.String("*-A*-A*-"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
    )
    EQ(
        python.String("*-A*-A*-"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
        -1,
    )
    EQ(
        python.String("*-A*-A*-"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
        sys.maxsize,
    )
    EQ(
        python.String("*-A*-A*-"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
        4,
    )
    EQ(
        python.String("*-A*-A*-"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
        3,
    )
    EQ(
        python.String("*-A*-A"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
        2,
    )
    EQ(
        python.String("*-AA"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
        1,
    )
    EQ(
        python.String("AA"),
        python.String("AA"),
        "replace",
        python.String(""),
        python.String("*-"),
        0,
    )

    # single character deletion (from==python.String("A"), to==python.String(""))
    EQ(
        python.String(""),
        python.String("A"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String(""),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String(""),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
        -1,
    )
    EQ(
        python.String(""),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
        sys.maxsize,
    )
    EQ(
        python.String(""),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
        4,
    )
    EQ(
        python.String(""),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
        3,
    )
    EQ(
        python.String("A"),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
        2,
    )
    EQ(
        python.String("AA"),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
        1,
    )
    EQ(
        python.String("AAA"),
        python.String("AAA"),
        "replace",
        python.String("A"),
        python.String(""),
        0,
    )
    EQ(
        python.String(""),
        python.String("AAAAAAAAAA"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String("BCD"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String("BCD"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        -1,
    )
    EQ(
        python.String("BCD"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        sys.maxsize,
    )
    EQ(
        python.String("BCD"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        5,
    )
    EQ(
        python.String("BCD"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        4,
    )
    EQ(
        python.String("BCDA"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        3,
    )
    EQ(
        python.String("BCADA"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        2,
    )
    EQ(
        python.String("BACADA"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        1,
    )
    EQ(
        python.String("ABACADA"),
        python.String("ABACADA"),
        "replace",
        python.String("A"),
        python.String(""),
        0,
    )
    EQ(
        python.String("BCD"),
        python.String("ABCAD"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String("BCD"),
        python.String("ABCADAA"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String("BCD"),
        python.String("BCD"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String("*************"),
        python.String("*************"),
        "replace",
        python.String("A"),
        python.String(""),
    )
    EQ(
        python.String("^A^"),
        python.String("^") + python.String("A") * 1000 + python.String("^"),
        "replace",
        python.String("A"),
        python.String(""),
        999,
    )

    # substring deletion (from==python.String("the"), to==python.String(""))
    EQ(
        python.String(""),
        python.String("the"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String("ater"),
        python.String("theater"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String(""),
        python.String("thethe"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String(""),
        python.String("thethethethe"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String("aaaa"),
        python.String("theatheatheathea"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String("that"),
        python.String("that"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String("thaet"),
        python.String("thaet"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String("here and re"),
        python.String("here and there"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String("here and re and re"),
        python.String("here and there and there"),
        "replace",
        python.String("the"),
        python.String(""),
        sys.maxsize,
    )
    EQ(
        python.String("here and re and re"),
        python.String("here and there and there"),
        "replace",
        python.String("the"),
        python.String(""),
        -1,
    )
    EQ(
        python.String("here and re and re"),
        python.String("here and there and there"),
        "replace",
        python.String("the"),
        python.String(""),
        3,
    )
    EQ(
        python.String("here and re and re"),
        python.String("here and there and there"),
        "replace",
        python.String("the"),
        python.String(""),
        2,
    )
    EQ(
        python.String("here and re and there"),
        python.String("here and there and there"),
        "replace",
        python.String("the"),
        python.String(""),
        1,
    )
    EQ(
        python.String("here and there and there"),
        python.String("here and there and there"),
        "replace",
        python.String("the"),
        python.String(""),
        0,
    )
    EQ(
        python.String("here and re and re"),
        python.String("here and there and there"),
        "replace",
        python.String("the"),
        python.String(""),
    )

    EQ(
        python.String("abc"),
        python.String("abc"),
        "replace",
        python.String("the"),
        python.String(""),
    )
    EQ(
        python.String("abcdefg"),
        python.String("abcdefg"),
        "replace",
        python.String("the"),
        python.String(""),
    )

    # substring deletion (from==python.String("bob"), to==python.String(""))
    EQ(
        python.String("bob"),
        python.String("bbobob"),
        "replace",
        python.String("bob"),
        python.String(""),
    )
    EQ(
        python.String("bobXbob"),
        python.String("bbobobXbbobob"),
        "replace",
        python.String("bob"),
        python.String(""),
    )
    EQ(
        python.String("aaaaaaa"),
        python.String("aaaaaaabob"),
        "replace",
        python.String("bob"),
        python.String(""),
    )
    EQ(
        python.String("aaaaaaa"),
        python.String("aaaaaaa"),
        "replace",
        python.String("bob"),
        python.String(""),
    )

    # single character replace in place (len(from)==len(to)==1)
    EQ(
        python.String("Who goes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("o"),
    )
    EQ(
        python.String("WhO gOes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("O"),
    )
    EQ(
        python.String("WhO gOes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("O"),
        sys.maxsize,
    )
    EQ(
        python.String("WhO gOes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("O"),
        -1,
    )
    EQ(
        python.String("WhO gOes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("O"),
        3,
    )
    EQ(
        python.String("WhO gOes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("O"),
        2,
    )
    EQ(
        python.String("WhO goes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("O"),
        1,
    )
    EQ(
        python.String("Who goes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("o"),
        python.String("O"),
        0,
    )

    EQ(
        python.String("Who goes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("a"),
        python.String("q"),
    )
    EQ(
        python.String("who goes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("W"),
        python.String("w"),
    )
    EQ(
        python.String("wwho goes there?ww"),
        python.String("WWho goes there?WW"),
        "replace",
        python.String("W"),
        python.String("w"),
    )
    EQ(
        python.String("Who goes there!"),
        python.String("Who goes there?"),
        "replace",
        python.String("?"),
        python.String("!"),
    )
    EQ(
        python.String("Who goes there!!"),
        python.String("Who goes there??"),
        "replace",
        python.String("?"),
        python.String("!"),
    )

    EQ(
        python.String("Who goes there?"),
        python.String("Who goes there?"),
        "replace",
        python.String("."),
        python.String("!"),
    )

    # substring replace in place (len(from)==len(to) > 1)
    EQ(
        python.String("Th** ** a t**sue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
    )
    EQ(
        python.String("Th** ** a t**sue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
        sys.maxsize,
    )
    EQ(
        python.String("Th** ** a t**sue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
        -1,
    )
    EQ(
        python.String("Th** ** a t**sue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
        4,
    )
    EQ(
        python.String("Th** ** a t**sue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
        3,
    )
    EQ(
        python.String("Th** ** a tissue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
        2,
    )
    EQ(
        python.String("Th** is a tissue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
        1,
    )
    EQ(
        python.String("This is a tissue"),
        python.String("This is a tissue"),
        "replace",
        python.String("is"),
        python.String("**"),
        0,
    )
    EQ(
        python.String("cobob"),
        python.String("bobob"),
        "replace",
        python.String("bob"),
        python.String("cob"),
    )
    EQ(
        python.String("cobobXcobocob"),
        python.String("bobobXbobobob"),
        "replace",
        python.String("bob"),
        python.String("cob"),
    )
    EQ(
        python.String("bobob"),
        python.String("bobob"),
        "replace",
        python.String("bot"),
        python.String("bot"),
    )

    # replace single character (len(from)==1, len(to)>1)
    EQ(
        python.String("ReyKKjaviKK"),
        python.String("Reykjavik"),
        "replace",
        python.String("k"),
        python.String("KK"),
    )
    EQ(
        python.String("ReyKKjaviKK"),
        python.String("Reykjavik"),
        "replace",
        python.String("k"),
        python.String("KK"),
        -1,
    )
    EQ(
        python.String("ReyKKjaviKK"),
        python.String("Reykjavik"),
        "replace",
        python.String("k"),
        python.String("KK"),
        sys.maxsize,
    )
    EQ(
        python.String("ReyKKjaviKK"),
        python.String("Reykjavik"),
        "replace",
        python.String("k"),
        python.String("KK"),
        2,
    )
    EQ(
        python.String("ReyKKjavik"),
        python.String("Reykjavik"),
        "replace",
        python.String("k"),
        python.String("KK"),
        1,
    )
    EQ(
        python.String("Reykjavik"),
        python.String("Reykjavik"),
        "replace",
        python.String("k"),
        python.String("KK"),
        0,
    )
    EQ(
        python.String("A----B----C----"),
        python.String("A.B.C."),
        "replace",
        python.String("."),
        python.String("----"),
    )
    # issue #15534
    EQ(
        python.String("...\u043c......&lt;"),
        python.String("...\u043c......<"),
        "replace",
        python.String("<"),
        python.String("&lt;"),
    )

    EQ(
        python.String("Reykjavik"),
        python.String("Reykjavik"),
        "replace",
        python.String("q"),
        python.String("KK"),
    )

    # replace substring (len(from)>1, len(to)!=len(from))
    EQ(
        python.String("ham, ham, eggs and ham"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
    )
    EQ(
        python.String("ham, ham, eggs and ham"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
        sys.maxsize,
    )
    EQ(
        python.String("ham, ham, eggs and ham"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
        -1,
    )
    EQ(
        python.String("ham, ham, eggs and ham"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
        4,
    )
    EQ(
        python.String("ham, ham, eggs and ham"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
        3,
    )
    EQ(
        python.String("ham, ham, eggs and spam"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
        2,
    )
    EQ(
        python.String("ham, spam, eggs and spam"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
        1,
    )
    EQ(
        python.String("spam, spam, eggs and spam"),
        python.String("spam, spam, eggs and spam"),
        "replace",
        python.String("spam"),
        python.String("ham"),
        0,
    )

    EQ(
        python.String("bobob"),
        python.String("bobobob"),
        "replace",
        python.String("bobob"),
        python.String("bob"),
    )
    EQ(
        python.String("bobobXbobob"),
        python.String("bobobobXbobobob"),
        "replace",
        python.String("bobob"),
        python.String("bob"),
    )
    EQ(
        python.String("BOBOBOB"),
        python.String("BOBOBOB"),
        "replace",
        python.String("bob"),
        python.String("bobby"),
    )

    checkequal(
        python.String("one@two!three!"),
        python.String("one!two!three!"),
        "replace",
        python.String("!"),
        python.String("@"),
        1,
    )
    checkequal(
        python.String("onetwothree"),
        python.String("one!two!three!"),
        "replace",
        python.String("!"),
        python.String(""),
    )
    checkequal(
        python.String("one@two@three!"),
        python.String("one!two!three!"),
        "replace",
        python.String("!"),
        python.String("@"),
        2,
    )
    checkequal(
        python.String("one@two@three@"),
        python.String("one!two!three!"),
        "replace",
        python.String("!"),
        python.String("@"),
        3,
    )
    checkequal(
        python.String("one@two@three@"),
        python.String("one!two!three!"),
        "replace",
        python.String("!"),
        python.String("@"),
        4,
    )
    checkequal(
        python.String("one!two!three!"),
        python.String("one!two!three!"),
        "replace",
        python.String("!"),
        python.String("@"),
        0,
    )
    checkequal(
        python.String("one@two@three@"),
        python.String("one!two!three!"),
        "replace",
        python.String("!"),
        python.String("@"),
    )
    checkequal(
        python.String("one!two!three!"),
        python.String("one!two!three!"),
        "replace",
        python.String("x"),
        python.String("@"),
    )
    checkequal(
        python.String("one!two!three!"),
        python.String("one!two!three!"),
        "replace",
        python.String("x"),
        python.String("@"),
        2,
    )
    checkequal(
        python.String("-a-b-c-"),
        python.String("abc"),
        "replace",
        python.String(""),
        python.String("-"),
    )
    checkequal(
        python.String("-a-b-c"),
        python.String("abc"),
        "replace",
        python.String(""),
        python.String("-"),
        3,
    )
    checkequal(
        python.String("abc"),
        python.String("abc"),
        "replace",
        python.String(""),
        python.String("-"),
        0,
    )
    checkequal(
        python.String(""),
        python.String(""),
        "replace",
        python.String(""),
        python.String(""),
    )
    checkequal(
        python.String("abc"),
        python.String("abc"),
        "replace",
        python.String("ab"),
        python.String("--"),
        0,
    )
    checkequal(
        python.String("abc"),
        python.String("abc"),
        "replace",
        python.String("xy"),
        python.String("--"),
    )
    # Next three for SF bug 422088: [OSF1 alpha] string.replace(); died with
    # MemoryError due to empty result (platform malloc issue when requesting
    # 0 bytes).
    checkequal(
        python.String(""),
        python.String("123"),
        "replace",
        python.String("123"),
        python.String(""),
    )
    checkequal(
        python.String(""),
        python.String("123123"),
        "replace",
        python.String("123"),
        python.String(""),
    )
    checkequal(
        python.String("x"),
        python.String("123x123"),
        "replace",
        python.String("123"),
        python.String(""),
    )

    checkraises(TypeError, python.String("hello"), "replace")
    checkraises(TypeError, python.String("hello"), "replace", 42)
    checkraises(TypeError, python.String("hello"), "replace", 42, python.String("h"))
    checkraises(TypeError, python.String("hello"), "replace", python.String("h"), 42)


def test_capitalize():
    checkequal(python.String(" hello "), python.String(" hello "), "capitalize")
    checkequal(python.String("Hello "), python.String("Hello "), "capitalize")
    checkequal(python.String("Hello "), python.String("hello "), "capitalize")
    checkequal(python.String("Aaaa"), python.String("aaaa"), "capitalize")
    checkequal(python.String("Aaaa"), python.String("AaAa"), "capitalize")

    checkraises(TypeError, python.String("hello"), "capitalize", 42)


def test_additional_split():
    checkequal(
        [
            python.String("this"),
            python.String("is"),
            python.String("the"),
            "split",
            python.String("function"),
        ],
        python.String("this is the split function"),
        "split",
    )

    # by whitespace
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d "),
        "split",
    )
    checkequal(
        [python.String("a"), python.String("b c d")],
        python.String("a b c d"),
        "split",
        None,
        1,
    )
    checkequal(
        [python.String("a"), python.String("b"), python.String("c d")],
        python.String("a b c d"),
        "split",
        None,
        2,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d"),
        "split",
        None,
        3,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d"),
        "split",
        None,
        4,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d"),
        "split",
        None,
        sys.maxsize - 1,
    )
    checkequal([python.String("a b c d")], python.String("a b c d"), "split", None, 0)
    checkequal([python.String("a b c d")], python.String("  a b c d"), "split", None, 0)
    checkequal(
        [python.String("a"), python.String("b"), python.String("c  d")],
        python.String("a  b  c  d"),
        "split",
        None,
        2,
    )

    checkequal([], python.String("         "), "split")
    checkequal([python.String("a")], python.String("  a    "), "split")
    checkequal(
        [python.String("a"), python.String("b")], python.String("  a    b   "), "split"
    )
    checkequal(
        [python.String("a"), python.String("b   ")],
        python.String("  a    b   "),
        "split",
        None,
        1,
    )
    checkequal(
        [python.String("a    b   c   ")],
        python.String("  a    b   c   "),
        "split",
        None,
        0,
    )
    checkequal(
        [python.String("a"), python.String("b   c   ")],
        python.String("  a    b   c   "),
        "split",
        None,
        1,
    )
    checkequal(
        [python.String("a"), python.String("b"), python.String("c   ")],
        python.String("  a    b   c   "),
        "split",
        None,
        2,
    )
    checkequal(
        [python.String("a"), python.String("b"), python.String("c")],
        python.String("  a    b   c   "),
        "split",
        None,
        3,
    )
    checkequal(
        [python.String("a"), python.String("b")],
        python.String("\n\ta \t\r b \v "),
        "split",
    )
    aaa = python.String(" a ") * 20
    checkequal([python.String("a")] * 20, aaa, "split")
    checkequal([python.String("a")] + [aaa[4:]], aaa, "split", None, 1)
    checkequal(
        [python.String("a")] * 19 + [python.String("a ")], aaa, "split", None, 19
    )

    for b in (
        python.String("arf\tbarf"),
        python.String("arf\nbarf"),
        python.String("arf\rbarf"),
        python.String("arf\fbarf"),
        python.String("arf\vbarf"),
    ):
        checkequal([python.String("arf"), python.String("barf")], b, "split")
        checkequal([python.String("arf"), python.String("barf")], b, "split", None)
        checkequal([python.String("arf"), python.String("barf")], b, "split", None, 2)


def test_additional_rsplit():
    checkequal(
        [
            python.String("this"),
            python.String("is"),
            python.String("the"),
            "rsplit",
            python.String("function"),
        ],
        python.String("this is the rsplit function"),
        "rsplit",
    )

    # by whitespace
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d "),
        "rsplit",
    )
    checkequal(
        [python.String("a b c"), python.String("d")],
        python.String("a b c d"),
        "rsplit",
        None,
        1,
    )
    checkequal(
        [python.String("a b"), python.String("c"), python.String("d")],
        python.String("a b c d"),
        "rsplit",
        None,
        2,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d"),
        "rsplit",
        None,
        3,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d"),
        "rsplit",
        None,
        4,
    )
    checkequal(
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
        python.String("a b c d"),
        "rsplit",
        None,
        sys.maxsize - 20,
    )
    checkequal([python.String("a b c d")], python.String("a b c d"), "rsplit", None, 0)
    checkequal(
        [python.String("a b c d")], python.String("a b c d  "), "rsplit", None, 0
    )
    checkequal(
        [python.String("a  b"), python.String("c"), python.String("d")],
        python.String("a  b  c  d"),
        "rsplit",
        None,
        2,
    )

    checkequal([], python.String("         "), "rsplit")
    checkequal([python.String("a")], python.String("  a    "), "rsplit")
    checkequal(
        [python.String("a"), python.String("b")], python.String("  a    b   "), "rsplit"
    )
    checkequal(
        [python.String("  a"), python.String("b")],
        python.String("  a    b   "),
        "rsplit",
        None,
        1,
    )
    checkequal(
        [python.String("  a    b   c")],
        python.String("  a    b   c   "),
        "rsplit",
        None,
        0,
    )
    checkequal(
        [python.String("  a    b"), python.String("c")],
        python.String("  a    b   c   "),
        "rsplit",
        None,
        1,
    )
    checkequal(
        [python.String("  a"), python.String("b"), python.String("c")],
        python.String("  a    b   c   "),
        "rsplit",
        None,
        2,
    )
    checkequal(
        [python.String("a"), python.String("b"), python.String("c")],
        python.String("  a    b   c   "),
        "rsplit",
        None,
        3,
    )
    checkequal(
        [python.String("a"), python.String("b")],
        python.String("\n\ta \t\r b \v "),
        "rsplit",
        None,
        88,
    )
    aaa = python.String(" a ") * 20
    checkequal([python.String("a")] * 20, aaa, "rsplit")
    checkequal([aaa[:-4]] + [python.String("a")], aaa, "rsplit", None, 1)
    checkequal(
        [python.String(" a  a")] + [python.String("a")] * 18, aaa, "rsplit", None, 18
    )

    for b in (
        python.String("arf\tbarf"),
        python.String("arf\nbarf"),
        python.String("arf\rbarf"),
        python.String("arf\fbarf"),
        python.String("arf\vbarf"),
    ):
        checkequal([python.String("arf"), python.String("barf")], b, "rsplit")
        checkequal([python.String("arf"), python.String("barf")], b, "rsplit", None)
        checkequal([python.String("arf"), python.String("barf")], b, "rsplit", None, 2)


def test_strip_whitespace():
    checkequal(python.String("hello"), python.String("   hello   "), "strip")
    checkequal(python.String("hello   "), python.String("   hello   "), "lstrip")
    checkequal(python.String("   hello"), python.String("   hello   "), "rstrip")
    checkequal(python.String("hello"), python.String("hello"), "strip")

    b = python.String(" \t\n\r\f\vabc \t\n\r\f\v")
    checkequal(python.String("abc"), b, "strip")
    checkequal(python.String("abc \t\n\r\f\v"), b, "lstrip")
    checkequal(python.String(" \t\n\r\f\vabc"), b, "rstrip")

    # strip/lstrip/rstrip with None arg
    checkequal(python.String("hello"), python.String("   hello   "), "strip", None)
    checkequal(python.String("hello   "), python.String("   hello   "), "lstrip", None)
    checkequal(python.String("   hello"), python.String("   hello   "), "rstrip", None)
    checkequal(python.String("hello"), python.String("hello"), "strip", None)


def test_strip():
    # strip/lstrip/rstrip with str arg
    checkequal(
        python.String("hello"),
        python.String("xyzzyhelloxyzzy"),
        "strip",
        python.String("xyz"),
    )
    checkequal(
        python.String("helloxyzzy"),
        python.String("xyzzyhelloxyzzy"),
        "lstrip",
        python.String("xyz"),
    )
    checkequal(
        python.String("xyzzyhello"),
        python.String("xyzzyhelloxyzzy"),
        "rstrip",
        python.String("xyz"),
    )
    checkequal(
        python.String("hello"), python.String("hello"), "strip", python.String("xyz")
    )
    checkequal(
        python.String(""),
        python.String("mississippi"),
        "strip",
        python.String("mississippi"),
    )

    # only trim the start and end; does not strip internal characters
    checkequal(
        python.String("mississipp"),
        python.String("mississippi"),
        "strip",
        python.String("i"),
    )

    checkraises(TypeError, python.String("hello"), "strip", 42, 42)
    checkraises(TypeError, python.String("hello"), "lstrip", 42, 42)
    checkraises(TypeError, python.String("hello"), "rstrip", 42, 42)


def test_ljust():
    checkequal(python.String("abc       "), python.String("abc"), "ljust", 10)
    checkequal(python.String("abc   "), python.String("abc"), "ljust", 6)
    checkequal(python.String("abc"), python.String("abc"), "ljust", 3)
    checkequal(python.String("abc"), python.String("abc"), "ljust", 2)
    checkequal(
        python.String("abc*******"),
        python.String("abc"),
        "ljust",
        10,
        python.String("*"),
    )
    checkraises(TypeError, python.String("abc"), "ljust")


def test_rjust():
    checkequal(python.String("       abc"), python.String("abc"), "rjust", 10)
    checkequal(python.String("   abc"), python.String("abc"), "rjust", 6)
    checkequal(python.String("abc"), python.String("abc"), "rjust", 3)
    checkequal(python.String("abc"), python.String("abc"), "rjust", 2)
    checkequal(
        python.String("*******abc"),
        python.String("abc"),
        "rjust",
        10,
        python.String("*"),
    )
    checkraises(TypeError, python.String("abc"), "rjust")


def test_center():
    checkequal(python.String("   abc    "), python.String("abc"), "center", 10)
    checkequal(python.String(" abc  "), python.String("abc"), "center", 6)
    checkequal(python.String("abc"), python.String("abc"), "center", 3)
    checkequal(python.String("abc"), python.String("abc"), "center", 2)
    checkequal(
        python.String("***abc****"),
        python.String("abc"),
        "center",
        10,
        python.String("*"),
    )
    checkraises(TypeError, python.String("abc"), "center")


def test_swapcase():
    checkequal(
        python.String("hEllO CoMPuTErS"), python.String("HeLLo cOmpUteRs"), "swapcase"
    )
    checkraises(TypeError, python.String("hello"), "swapcase", 42)


def test_zfill():
    checkequal(python.String("123"), python.String("123"), "zfill", 2)
    checkequal(python.String("123"), python.String("123"), "zfill", 3)
    checkequal(python.String("0123"), python.String("123"), "zfill", 4)
    checkequal(python.String("+123"), python.String("+123"), "zfill", 3)
    checkequal(python.String("+123"), python.String("+123"), "zfill", 4)
    checkequal(python.String("+0123"), python.String("+123"), "zfill", 5)
    checkequal(python.String("-123"), python.String("-123"), "zfill", 3)
    checkequal(python.String("-123"), python.String("-123"), "zfill", 4)
    checkequal(python.String("-0123"), python.String("-123"), "zfill", 5)
    checkequal(python.String("000"), python.String(""), "zfill", 3)
    checkequal(python.String("34"), python.String("34"), "zfill", 1)
    checkequal(python.String("0034"), python.String("34"), "zfill", 4)

    checkraises(TypeError, python.String("123"), "zfill")


def test_islower():
    checkequal(False, python.String(""), "islower")
    checkequal(True, python.String("a"), "islower")
    checkequal(False, python.String("A"), "islower")
    checkequal(False, python.String("\n"), "islower")
    checkequal(True, python.String("abc"), "islower")
    checkequal(False, python.String("aBc"), "islower")
    checkequal(True, python.String("abc\n"), "islower")
    checkraises(TypeError, python.String("abc"), "islower", 42)


def test_isupper():
    checkequal(False, python.String(""), "isupper")
    checkequal(False, python.String("a"), "isupper")
    checkequal(True, python.String("A"), "isupper")
    checkequal(False, python.String("\n"), "isupper")
    checkequal(True, python.String("ABC"), "isupper")
    checkequal(False, python.String("AbC"), "isupper")
    checkequal(True, python.String("ABC\n"), "isupper")
    checkraises(TypeError, python.String("abc"), "isupper", 42)


def test_istitle():
    checkequal(False, python.String(""), "istitle")
    checkequal(False, python.String("a"), "istitle")
    checkequal(True, python.String("A"), "istitle")
    checkequal(False, python.String("\n"), "istitle")
    checkequal(True, python.String("A Titlecased Line"), "istitle")
    checkequal(True, python.String("A\nTitlecased Line"), "istitle")
    checkequal(True, python.String("A Titlecased, Line"), "istitle")
    checkequal(False, python.String("Not a capitalized String"), "istitle")
    checkequal(False, python.String("Not\ta Titlecase String"), "istitle")
    checkequal(False, python.String("Not--a Titlecase String"), "istitle")
    checkequal(False, python.String("NOT"), "istitle")
    checkraises(TypeError, python.String("abc"), "istitle", 42)


def test_isspace():
    checkequal(False, python.String(""), "isspace")
    checkequal(False, python.String("a"), "isspace")
    checkequal(True, python.String(" "), "isspace")
    checkequal(True, python.String("\t"), "isspace")
    checkequal(True, python.String("\r"), "isspace")
    checkequal(True, python.String("\n"), "isspace")
    checkequal(True, python.String(" \t\r\n"), "isspace")
    checkequal(False, python.String(" \t\r\na"), "isspace")
    checkraises(TypeError, python.String("abc"), "isspace", 42)


def test_isalpha():
    checkequal(False, python.String(""), "isalpha")
    checkequal(True, python.String("a"), "isalpha")
    checkequal(True, python.String("A"), "isalpha")
    checkequal(False, python.String("\n"), "isalpha")
    checkequal(True, python.String("abc"), "isalpha")
    checkequal(False, python.String("aBc123"), "isalpha")
    checkequal(False, python.String("abc\n"), "isalpha")
    checkraises(TypeError, python.String("abc"), "isalpha", 42)


def test_isalnum():
    checkequal(False, python.String(""), "isalnum")
    checkequal(True, python.String("a"), "isalnum")
    checkequal(True, python.String("A"), "isalnum")
    checkequal(False, python.String("\n"), "isalnum")
    checkequal(True, python.String("123abc456"), "isalnum")
    checkequal(True, python.String("a1b3c"), "isalnum")
    checkequal(False, python.String("aBc000 "), "isalnum")
    checkequal(False, python.String("abc\n"), "isalnum")
    checkraises(TypeError, python.String("abc"), "isalnum", 42)


def test_isascii():
    if sys.version_info >= (3, 7):
        checkequal(True, python.String(""), "isascii")
        checkequal(True, python.String("\x00"), "isascii")
        checkequal(True, python.String("\x7f"), "isascii")
        checkequal(True, python.String("\x00\x7f"), "isascii")
        checkequal(False, python.String("\x80"), "isascii")
        checkequal(False, python.String("\xe9"), "isascii")
        # bytes.isascii() and bytearray.isascii() has optimization which
        # check 4 or 8 bytes at once.  So check some alignments.
        for p in range(8):
            checkequal(True, python.String(" ") * p + python.String("\x7f"), "isascii")
            checkequal(False, python.String(" ") * p + python.String("\x80"), "isascii")
            checkequal(
                True,
                python.String(" ") * p + python.String("\x7f") + python.String(" ") * 8,
                "isascii",
            )
            checkequal(
                False,
                python.String(" ") * p + python.String("\x80") + python.String(" ") * 8,
                "isascii",
            )
    else:
        with pytest.raises(AttributeError):
            checkequal(True, python.String(""), "isascii")


def test_isdigit():
    checkequal(False, python.String(""), "isdigit")
    checkequal(False, python.String("a"), "isdigit")
    checkequal(True, python.String("0"), "isdigit")
    checkequal(True, python.String("0123456789"), "isdigit")
    checkequal(False, python.String("0123456789a"), "isdigit")

    checkraises(TypeError, python.String("abc"), "isdigit", 42)


def test_title():
    checkequal(python.String(" Hello "), python.String(" hello "), "title")
    checkequal(python.String("Hello "), python.String("hello "), "title")
    checkequal(python.String("Hello "), python.String("Hello "), "title")
    checkequal(
        python.String("Format This As Title String"),
        python.String("fOrMaT thIs aS titLe String"),
        "title",
    )
    checkequal(
        python.String("Format,This-As*Title;String"),
        python.String("fOrMaT,thIs-aS*titLe;String"),
        "title",
    )
    checkequal(python.String("Getint"), python.String("getInt"), "title")
    checkraises(TypeError, python.String("hello"), "title", 42)


def test_splitlines():
    checkequal(
        [
            python.String("abc"),
            python.String("def"),
            python.String(""),
            python.String("ghi"),
        ],
        python.String("abc\ndef\n\rghi"),
        "splitlines",
    )
    checkequal(
        [
            python.String("abc"),
            python.String("def"),
            python.String(""),
            python.String("ghi"),
        ],
        python.String("abc\ndef\n\r\nghi"),
        "splitlines",
    )
    checkequal(
        [python.String("abc"), python.String("def"), python.String("ghi")],
        python.String("abc\ndef\r\nghi"),
        "splitlines",
    )
    checkequal(
        [python.String("abc"), python.String("def"), python.String("ghi")],
        python.String("abc\ndef\r\nghi\n"),
        "splitlines",
    )
    checkequal(
        [
            python.String("abc"),
            python.String("def"),
            python.String("ghi"),
            python.String(""),
        ],
        python.String("abc\ndef\r\nghi\n\r"),
        "splitlines",
    )
    checkequal(
        [
            python.String(""),
            python.String("abc"),
            python.String("def"),
            python.String("ghi"),
            python.String(""),
        ],
        python.String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
    )
    checkequal(
        [
            python.String(""),
            python.String("abc"),
            python.String("def"),
            python.String("ghi"),
            python.String(""),
        ],
        python.String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        False,
    )
    checkequal(
        [
            python.String("\n"),
            python.String("abc\n"),
            python.String("def\r\n"),
            python.String("ghi\n"),
            python.String("\r"),
        ],
        python.String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        True,
    )
    checkequal(
        [
            python.String(""),
            python.String("abc"),
            python.String("def"),
            python.String("ghi"),
            python.String(""),
        ],
        python.String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        keepends=False,
    )
    checkequal(
        [
            python.String("\n"),
            python.String("abc\n"),
            python.String("def\r\n"),
            python.String("ghi\n"),
            python.String("\r"),
        ],
        python.String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        keepends=True,
    )

    checkraises(TypeError, python.String("abc"), "splitlines", 42, 42)


# fixes python <= 3.7
# https://github.com/python/cpython/commit/b015fc86f7b1f35283804bfee788cce0a5495df7
def test_capitalize_nonascii():
    # check that titlecased chars are lowered correctly
    # \u1ffc is the titlecased char
    # \u03a9\u0399
    if sys.version_info >= (3, 8):
        # a, b, capitalize
        # ῼῳῳῳ, ῳῳῼῼ, capitalize
        checkequal(
            python.String("\u1ffc\u1ff3\u1ff3\u1ff3"),
            python.String("\u1ff3\u1ff3\u1ffc\u1ffc"),
            "capitalize",
        )
    else:
        # a, b, capitalize
        # ΩΙῳῳῳ, ῳῳῼῼ, capitalize
        checkequal(
            python.String("\u03a9\u0399\u1ff3\u1ff3\u1ff3"),
            python.String("\u1ff3\u1ff3\u1ffc\u1ffc"),
            "capitalize",
        )
    # check with cased non-letter chars
    checkequal(
        python.String("\u24c5\u24e8\u24e3\u24d7\u24de\u24dd"),
        python.String("\u24c5\u24ce\u24c9\u24bd\u24c4\u24c3"),
        "capitalize",
    )
    checkequal(
        python.String("\u24c5\u24e8\u24e3\u24d7\u24de\u24dd"),
        python.String("\u24df\u24e8\u24e3\u24d7\u24de\u24dd"),
        "capitalize",
    )
    checkequal(
        python.String("\u2160\u2171\u2172"),
        python.String("\u2160\u2161\u2162"),
        "capitalize",
    )
    checkequal(
        python.String("\u2160\u2171\u2172"),
        python.String("\u2170\u2171\u2172"),
        "capitalize",
    )
    # check with Ll chars with no upper - nothing changes here
    checkequal(
        python.String("\u019b\u1d00\u1d86\u0221\u1fb7"),
        python.String("\u019b\u1d00\u1d86\u0221\u1fb7"),
        "capitalize",
    )


def test_startswith():
    checkequal(True, python.String("hello"), "startswith", python.String("he"))
    checkequal(True, python.String("hello"), "startswith", python.String("hello"))
    checkequal(
        False, python.String("hello"), "startswith", python.String("hello world")
    )
    checkequal(True, python.String("hello"), "startswith", python.String(""))
    checkequal(False, python.String("hello"), "startswith", python.String("ello"))
    checkequal(True, python.String("hello"), "startswith", python.String("ello"), 1)
    checkequal(True, python.String("hello"), "startswith", python.String("o"), 4)
    checkequal(False, python.String("hello"), "startswith", python.String("o"), 5)
    checkequal(True, python.String("hello"), "startswith", python.String(""), 5)
    checkequal(False, python.String("hello"), "startswith", python.String("lo"), 6)
    checkequal(
        True, python.String("helloworld"), "startswith", python.String("lowo"), 3
    )
    checkequal(
        True, python.String("helloworld"), "startswith", python.String("lowo"), 3, 7
    )
    checkequal(
        False, python.String("helloworld"), "startswith", python.String("lowo"), 3, 6
    )
    checkequal(True, python.String(""), "startswith", python.String(""), 0, 1)
    checkequal(True, python.String(""), "startswith", python.String(""), 0, 0)
    checkequal(False, python.String(""), "startswith", python.String(""), 1, 0)

    # test negative indices
    checkequal(True, python.String("hello"), "startswith", python.String("he"), 0, -1)
    checkequal(True, python.String("hello"), "startswith", python.String("he"), -53, -1)
    checkequal(
        False, python.String("hello"), "startswith", python.String("hello"), 0, -1
    )
    checkequal(
        False,
        python.String("hello"),
        "startswith",
        python.String("hello world"),
        -1,
        -10,
    )
    checkequal(False, python.String("hello"), "startswith", python.String("ello"), -5)
    checkequal(True, python.String("hello"), "startswith", python.String("ello"), -4)
    checkequal(False, python.String("hello"), "startswith", python.String("o"), -2)
    checkequal(True, python.String("hello"), "startswith", python.String("o"), -1)
    checkequal(True, python.String("hello"), "startswith", python.String(""), -3, -3)
    checkequal(False, python.String("hello"), "startswith", python.String("lo"), -9)

    checkraises(TypeError, python.String("hello"), "startswith")
    checkraises(TypeError, python.String("hello"), "startswith", 42)

    # test tuple arguments
    checkequal(
        True,
        python.String("hello"),
        "startswith",
        (python.String("he"), python.String("ha")),
    )
    checkequal(
        False,
        python.String("hello"),
        "startswith",
        (python.String("lo"), python.String("llo")),
    )
    checkequal(
        True,
        python.String("hello"),
        "startswith",
        (python.String("hellox"), python.String("hello")),
    )
    checkequal(False, python.String("hello"), "startswith", ())
    checkequal(
        True,
        python.String("helloworld"),
        "startswith",
        (python.String("hellowo"), python.String("rld"), python.String("lowo")),
        3,
    )
    checkequal(
        False,
        python.String("helloworld"),
        "startswith",
        (python.String("hellowo"), python.String("ello"), python.String("rld")),
        3,
    )
    checkequal(
        True,
        python.String("hello"),
        "startswith",
        (python.String("lo"), python.String("he")),
        0,
        -1,
    )
    checkequal(
        False,
        python.String("hello"),
        "startswith",
        (python.String("he"), python.String("hel")),
        0,
        1,
    )
    checkequal(
        True,
        python.String("hello"),
        "startswith",
        (python.String("he"), python.String("hel")),
        0,
        2,
    )

    checkraises(TypeError, python.String("hello"), "startswith", (42,))


def test_endswith():
    checkequal(True, python.String("hello"), "endswith", python.String("lo"))
    checkequal(False, python.String("hello"), "endswith", python.String("he"))
    checkequal(True, python.String("hello"), "endswith", python.String(""))
    checkequal(False, python.String("hello"), "endswith", python.String("hello world"))
    checkequal(False, python.String("helloworld"), "endswith", python.String("worl"))
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("worl"), 3, 9
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("world"), 3, 12
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("lowo"), 1, 7
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("lowo"), 2, 7
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("lowo"), 3, 7
    )
    checkequal(
        False, python.String("helloworld"), "endswith", python.String("lowo"), 4, 7
    )
    checkequal(
        False, python.String("helloworld"), "endswith", python.String("lowo"), 3, 8
    )
    checkequal(False, python.String("ab"), "endswith", python.String("ab"), 0, 1)
    checkequal(False, python.String("ab"), "endswith", python.String("ab"), 0, 0)
    checkequal(True, python.String(""), "endswith", python.String(""), 0, 1)
    checkequal(True, python.String(""), "endswith", python.String(""), 0, 0)
    checkequal(False, python.String(""), "endswith", python.String(""), 1, 0)

    # test negative indices
    checkequal(True, python.String("hello"), "endswith", python.String("lo"), -2)
    checkequal(False, python.String("hello"), "endswith", python.String("he"), -2)
    checkequal(True, python.String("hello"), "endswith", python.String(""), -3, -3)
    checkequal(
        False, python.String("hello"), "endswith", python.String("hello world"), -10, -2
    )
    checkequal(
        False, python.String("helloworld"), "endswith", python.String("worl"), -6
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("worl"), -5, -1
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("worl"), -5, 9
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("world"), -7, 12
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("lowo"), -99, -3
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("lowo"), -8, -3
    )
    checkequal(
        True, python.String("helloworld"), "endswith", python.String("lowo"), -7, -3
    )
    checkequal(
        False, python.String("helloworld"), "endswith", python.String("lowo"), 3, -4
    )
    checkequal(
        False, python.String("helloworld"), "endswith", python.String("lowo"), -8, -2
    )

    checkraises(TypeError, python.String("hello"), "endswith")
    checkraises(TypeError, python.String("hello"), "endswith", 42)

    # test tuple arguments
    checkequal(
        False,
        python.String("hello"),
        "endswith",
        (python.String("he"), python.String("ha")),
    )
    checkequal(
        True,
        python.String("hello"),
        "endswith",
        (python.String("lo"), python.String("llo")),
    )
    checkequal(
        True,
        python.String("hello"),
        "endswith",
        (python.String("hellox"), python.String("hello")),
    )
    checkequal(False, python.String("hello"), "endswith", ())
    checkequal(
        True,
        python.String("helloworld"),
        "endswith",
        (python.String("hellowo"), python.String("rld"), python.String("lowo")),
        3,
    )
    checkequal(
        False,
        python.String("helloworld"),
        "endswith",
        (python.String("hellowo"), python.String("ello"), python.String("rld")),
        3,
        -1,
    )
    checkequal(
        True,
        python.String("hello"),
        "endswith",
        (python.String("hell"), python.String("ell")),
        0,
        -1,
    )
    checkequal(
        False,
        python.String("hello"),
        "endswith",
        (python.String("he"), python.String("hel")),
        0,
        1,
    )
    checkequal(
        True,
        python.String("hello"),
        "endswith",
        (python.String("he"), python.String("hell")),
        0,
        4,
    )

    checkraises(TypeError, python.String("hello"), "endswith", (42,))


def test___contains__():
    checkequal(True, python.String(""), "__contains__", python.String(""))
    checkequal(True, python.String("abc"), "__contains__", python.String(""))
    checkequal(False, python.String("abc"), "__contains__", python.String("\0"))
    checkequal(True, python.String("\0abc"), "__contains__", python.String("\0"))
    checkequal(True, python.String("abc\0"), "__contains__", python.String("\0"))
    checkequal(True, python.String("\0abc"), "__contains__", python.String("a"))
    checkequal(True, python.String("asdf"), "__contains__", python.String("asdf"))
    checkequal(False, python.String("asd"), "__contains__", python.String("asdf"))
    checkequal(False, python.String(""), "__contains__", python.String("asdf"))


def test_subscript():
    checkequal(python.String("a"), python.String("abc"), "__getitem__", 0)
    checkequal(python.String("c"), python.String("abc"), "__getitem__", -1)
    checkequal(python.String("a"), python.String("abc"), "__getitem__", 0)
    checkequal(python.String("abc"), python.String("abc"), "__getitem__", slice(0, 3))
    checkequal(
        python.String("abc"), python.String("abc"), "__getitem__", slice(0, 1000)
    )
    checkequal(python.String("a"), python.String("abc"), "__getitem__", slice(0, 1))
    checkequal(python.String(""), python.String("abc"), "__getitem__", slice(0, 0))

    checkraises(TypeError, python.String("abc"), "__getitem__", python.String("def"))


def test_slice():
    checkequal(
        python.String("abc"), python.String("abc"), "__getitem__", slice(0, 1000)
    )
    checkequal(python.String("abc"), python.String("abc"), "__getitem__", slice(0, 3))
    checkequal(python.String("ab"), python.String("abc"), "__getitem__", slice(0, 2))
    checkequal(python.String("bc"), python.String("abc"), "__getitem__", slice(1, 3))
    checkequal(python.String("b"), python.String("abc"), "__getitem__", slice(1, 2))
    checkequal(python.String(""), python.String("abc"), "__getitem__", slice(2, 2))
    checkequal(
        python.String(""), python.String("abc"), "__getitem__", slice(1000, 1000)
    )
    checkequal(
        python.String(""), python.String("abc"), "__getitem__", slice(2000, 1000)
    )
    checkequal(python.String(""), python.String("abc"), "__getitem__", slice(2, 1))

    checkraises(TypeError, python.String("abc"), "__getitem__", python.String("def"))


def test_extended_getslice():
    # Test extended slicing by comparing with list slicing.
    s = string.ascii_letters + string.digits
    indices = (0, None, 1, 3, 41, sys.maxsize, -1, -2, -37)
    for start in indices:
        for stop in indices:
            # Skip step 0 (invalid)
            for step in indices[1:]:
                L = list(s)[start:stop:step]
                checkequal(
                    python.String("").join(L),
                    s,
                    "__getitem__",
                    slice(start, stop, step),
                )


def test_mul():
    checkequal(python.String(""), python.String("abc"), "__mul__", -1)
    checkequal(python.String(""), python.String("abc"), "__mul__", 0)
    checkequal(python.String("abc"), python.String("abc"), "__mul__", 1)
    checkequal(python.String("abcabcabc"), python.String("abc"), "__mul__", 3)
    checkraises(TypeError, python.String("abc"), "__mul__")
    checkraises(TypeError, python.String("abc"), "__mul__", python.String(""))
    # XXX: on a 64-bit system, this doesn't raise an overflow error,
    # but either raises a MemoryError, or succeeds (if you have 54TiB)
    # checkraises(OverflowError, 10000*'abc', '__mul__', 2000000000)


@pytest.mark.slow
def test_join():
    checkequal(
        python.String("a b c d"),
        python.String(" "),
        "join",
        [
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ],
    )
    checkequal(
        python.String("abcd"),
        python.String(""),
        "join",
        (
            python.String("a"),
            python.String("b"),
            python.String("c"),
            python.String("d"),
        ),
    )
    checkequal(
        python.String("bd"),
        python.String(""),
        "join",
        (python.String(""), python.String("b"), python.String(""), python.String("d")),
    )
    checkequal(
        python.String("ac"),
        python.String(""),
        "join",
        (python.String("a"), python.String(""), python.String("c"), python.String("")),
    )
    checkequal(python.String("w x y z"), python.String(" "), "join", Sequence())
    checkequal(
        python.String("abc"), python.String("a"), "join", (python.String("abc"),)
    )
    checkequal(
        python.String("z"),
        python.String("a"),
        "join",
        python.List([python.String("z")]),
    )
    checkequal(
        python.String("a.b.c"),
        python.String("."),
        "join",
        [python.String("a"), python.String("b"), python.String("c")],
    )
    checkraises(
        TypeError,
        python.String("."),
        "join",
        [python.String("a"), python.String("b"), 3],
    )
    for i in [5, 25, 125]:
        checkequal(
            (((python.String("a") * i) + python.String("-")) * i)[:-1],
            python.String("-"),
            "join",
            [python.String("a") * i] * i,
        )
        checkequal(
            (((python.String("a") * i) + python.String("-")) * i)[:-1],
            python.String("-"),
            "join",
            (python.String("a") * i,) * i,
        )

    checkequal(python.String("a b c"), python.String(" "), "join", BadSeq2())

    checkraises(TypeError, python.String(" "), "join")
    checkraises(TypeError, python.String(" "), "join", None)
    checkraises(TypeError, python.String(" "), "join", 7)
    checkraises(TypeError, python.String(" "), "join", [1, 2, bytes()])


def test_formatting():
    checkequal(
        python.String("+hello+"),
        python.String("+%s+"),
        "__mod__",
        python.String("hello"),
    )
    checkequal(python.String("+10+"), python.String("+%d+"), "__mod__", 10)
    checkequal(python.String("a"), python.String("%c"), "__mod__", python.String("a"))
    checkequal(python.String("a"), python.String("%c"), "__mod__", python.String("a"))
    checkequal(python.String("$"), python.String("%c"), "__mod__", 36)
    checkequal(python.String("10"), python.String("%d"), "__mod__", 10)
    checkequal(python.String("\x7f"), python.String("%c"), "__mod__", 0x7F)

    for ordinal in (-100, 0x200000):
        # unicode raises ValueError, str raises OverflowError
        checkraises(
            (ValueError, OverflowError), python.String("%c"), "__mod__", ordinal
        )

    longvalue = sys.maxsize + 10
    slongvalue = str(longvalue)
    checkequal(python.String(" 42"), python.String("%3ld"), "__mod__", 42)
    checkequal(python.String("42"), python.String("%d"), "__mod__", 42.0)
    checkequal(slongvalue, python.String("%d"), "__mod__", longvalue)
    checkcall(python.String("%d"), "__mod__", float(longvalue))
    checkequal(python.String("0042.00"), python.String("%07.2f"), "__mod__", 42)
    checkequal(python.String("0042.00"), python.String("%07.2F"), "__mod__", 42)

    checkraises(TypeError, python.String("abc"), "__mod__")
    checkraises(TypeError, python.String("%(foo)s"), "__mod__", 42)
    checkraises(TypeError, python.String("%s%s"), "__mod__", (42,))
    checkraises(TypeError, python.String("%c"), "__mod__", (None,))
    checkraises(ValueError, python.String("%(foo"), "__mod__", {})
    checkraises(
        TypeError,
        python.String("%(foo)s %(bar)s"),
        "__mod__",
        (python.String("foo"), 42),
    )
    checkraises(
        TypeError, python.String("%d"), "__mod__", python.String("42")
    )  # not numeric
    checkraises(
        TypeError, python.String("%d"), "__mod__", (42 + 0j)
    )  # no int conversion provided

    # argument names with properly nested brackets are supported
    checkequal(
        python.String("bar"),
        python.String("%((foo))s"),
        "__mod__",
        {python.String("(foo)"): python.String("bar")},
    )

    # 100 is a magic number in PyUnicode_Format, this forces a resize
    checkequal(
        103 * python.String("a") + python.String("x"),
        python.String("%sx"),
        "__mod__",
        103 * python.String("a"),
    )

    checkraises(
        TypeError,
        python.String("%*s"),
        "__mod__",
        (python.String("foo"), python.String("bar")),
    )
    checkraises(
        TypeError, python.String("%10.*f"), "__mod__", (python.String("foo"), 42.0)
    )
    checkraises(ValueError, python.String("%10"), "__mod__", (42,))

    # Outrageously large width or precision should raise ValueError.
    checkraises(ValueError, python.String("%%%df") % (2 ** 64), "__mod__", (3.2))
    checkraises(ValueError, python.String("%%.%df") % (2 ** 64), "__mod__", (3.2))
    checkraises(
        OverflowError,
        python.String("%*s"),
        "__mod__",
        (sys.maxsize + 1, python.String("")),
    )
    checkraises(
        OverflowError, python.String("%.*f"), "__mod__", (sys.maxsize + 1, 1.0 / 7)
    )

    class X(object):
        pass

    checkraises(TypeError, python.String("abc"), "__mod__", X())


@support.cpython_only
def test_formatting_c_limits():
    # third party
    from _testcapi import INT_MAX
    from _testcapi import PY_SSIZE_T_MAX
    from _testcapi import UINT_MAX

    SIZE_MAX = (1 << (PY_SSIZE_T_MAX.bit_length() + 1)) - 1
    checkraises(
        OverflowError,
        python.String("%*s"),
        "__mod__",
        (PY_SSIZE_T_MAX + 1, python.String("")),
    )
    checkraises(OverflowError, python.String("%.*f"), "__mod__", (INT_MAX + 1, 1.0 / 7))
    # Issue 15989
    checkraises(
        OverflowError,
        python.String("%*s"),
        "__mod__",
        (SIZE_MAX + 1, python.String("")),
    )
    checkraises(
        OverflowError, python.String("%.*f"), "__mod__", (UINT_MAX + 1, 1.0 / 7)
    )


@pytest.mark.slow
def test_floatformatting():
    # float formatting
    for prec in range(100):
        format = python.String("%%.%if") % prec
        value = 0.01
        for x in range(60):
            value = value * 3.14159265359 / 3.0 * 10.0
            checkcall(format, "__mod__", value)


def test_inplace_rewrites():
    # Check that strings don't copy and modify cached single-character strings
    checkequal(python.String("a"), python.String("A"), "lower")
    checkequal(True, python.String("A"), "isupper")
    checkequal(python.String("A"), python.String("a"), "upper")
    checkequal(True, python.String("a"), "islower")

    checkequal(
        python.String("a"),
        python.String("A"),
        "replace",
        python.String("A"),
        python.String("a"),
    )
    checkequal(True, python.String("A"), "isupper")

    checkequal(python.String("A"), python.String("a"), "capitalize")
    checkequal(True, python.String("a"), "islower")

    checkequal(python.String("A"), python.String("a"), "swapcase")
    checkequal(True, python.String("a"), "islower")

    checkequal(python.String("A"), python.String("a"), "title")
    checkequal(True, python.String("a"), "islower")


def test_partition():
    checkequal(
        (
            python.String("this is the par"),
            python.String("ti"),
            python.String("tion method"),
        ),
        python.String("this is the partition method"),
        "partition",
        python.String("ti"),
    )

    # from raymond's original specification
    S = python.String("http://www.python.org")
    checkequal(
        (python.String("http"), python.String("://"), python.String("www.python.org")),
        S,
        "partition",
        python.String("://"),
    )
    checkequal(
        (python.String("http://www.python.org"), python.String(""), python.String("")),
        S,
        "partition",
        python.String("?"),
    )
    checkequal(
        (python.String(""), python.String("http://"), python.String("www.python.org")),
        S,
        "partition",
        python.String("http://"),
    )
    checkequal(
        (python.String("http://www.python."), python.String("org"), python.String("")),
        S,
        "partition",
        python.String("org"),
    )

    checkraises(ValueError, S, "partition", python.String(""))
    checkraises(TypeError, S, "partition", None)


def test_rpartition():
    checkequal(
        (
            python.String("this is the rparti"),
            python.String("ti"),
            python.String("on method"),
        ),
        python.String("this is the rpartition method"),
        "rpartition",
        python.String("ti"),
    )

    # from raymond's original specification
    S = python.String("http://www.python.org")
    checkequal(
        (python.String("http"), python.String("://"), python.String("www.python.org")),
        S,
        "rpartition",
        python.String("://"),
    )
    checkequal(
        (python.String(""), python.String(""), python.String("http://www.python.org")),
        S,
        "rpartition",
        python.String("?"),
    )
    checkequal(
        (python.String(""), python.String("http://"), python.String("www.python.org")),
        S,
        "rpartition",
        python.String("http://"),
    )
    checkequal(
        (python.String("http://www.python."), python.String("org"), python.String("")),
        S,
        "rpartition",
        python.String("org"),
    )

    checkraises(ValueError, S, "rpartition", python.String(""))
    checkraises(TypeError, S, "rpartition", None)


def test_none_arguments():
    # issue 11828
    s = python.String("hello")
    checkequal(2, s, "find", python.String("l"), None)
    checkequal(3, s, "find", python.String("l"), -2, None)
    checkequal(2, s, "find", python.String("l"), None, -2)
    checkequal(0, s, "find", python.String("h"), None, None)

    checkequal(3, s, "rfind", python.String("l"), None)
    checkequal(3, s, "rfind", python.String("l"), -2, None)
    checkequal(2, s, "rfind", python.String("l"), None, -2)
    checkequal(0, s, "rfind", python.String("h"), None, None)

    checkequal(2, s, "index", python.String("l"), None)
    checkequal(3, s, "index", python.String("l"), -2, None)
    checkequal(2, s, "index", python.String("l"), None, -2)
    checkequal(0, s, "index", python.String("h"), None, None)

    checkequal(3, s, "rindex", python.String("l"), None)
    checkequal(3, s, "rindex", python.String("l"), -2, None)
    checkequal(2, s, "rindex", python.String("l"), None, -2)
    checkequal(0, s, "rindex", python.String("h"), None, None)

    checkequal(2, s, "count", python.String("l"), None)
    checkequal(1, s, "count", python.String("l"), -2, None)
    checkequal(1, s, "count", python.String("l"), None, -2)
    checkequal(0, s, "count", python.String("x"), None, None)

    checkequal(True, s, "endswith", python.String("o"), None)
    checkequal(True, s, "endswith", python.String("lo"), -2, None)
    checkequal(True, s, "endswith", python.String("l"), None, -2)
    checkequal(False, s, "endswith", python.String("x"), None, None)

    checkequal(True, s, "startswith", python.String("h"), None)
    checkequal(True, s, "startswith", python.String("l"), -2, None)
    checkequal(True, s, "startswith", python.String("h"), None, -2)
    checkequal(False, s, "startswith", python.String("x"), None, None)
