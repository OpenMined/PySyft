# flake8: noqa
"""
File copied from cpython test suite:
https://github.com/python/cpython/blob/3.9/Lib/test/string_tests.py

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
from syft.lib.python import List
from syft.lib.python import String


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
    checkequal(0, String("abcdefghiabc"), "find", String("abc"))
    checkequal(9, String("abcdefghiabc"), "find", String("abc"), 1)
    checkequal(-1, String("abcdefghiabc"), "find", String("def"), 4)

    checkequal(0, String("abc"), "find", String(""), 0)
    checkequal(3, String("abc"), "find", String(""), 3)
    checkequal(-1, String("abc"), "find", String(""), 4)

    # to check the ability to pass None as defaults
    checkequal(2, String("rrarrrrrrrrra"), "find", String("a"))
    checkequal(12, String("rrarrrrrrrrra"), "find", String("a"), 4)
    checkequal(-1, String("rrarrrrrrrrra"), "find", String("a"), 4, 6)
    checkequal(12, String("rrarrrrrrrrra"), "find", String("a"), 4, None)
    checkequal(2, String("rrarrrrrrrrra"), "find", String("a"), None, 6)

    checkraises(TypeError, String("hello"), "find")

    if contains_bytes:
        checkequal(-1, String("hello"), "find", 42)
    else:
        checkraises(TypeError, String("hello"), "find", 42)

    checkequal(0, String(""), "find", String(""))
    checkequal(-1, String(""), "find", String(""), 1, 1)
    checkequal(-1, String(""), "find", String(""), sys.maxsize, 0)

    checkequal(-1, String(""), "find", String("xx"))
    checkequal(-1, String(""), "find", String("xx"), 1, 1)
    checkequal(-1, String(""), "find", String("xx"), sys.maxsize, 0)

    # issue 7458
    checkequal(-1, String("ab"), "find", String("xxx"), sys.maxsize + 1, 0)

    # For a variety of combinations,
    #    verify that str.find() matches __contains__
    #    and that the found substring is really at that location
    charset = [String(""), String("a"), String("b")]
    digits = 4
    base = len(charset)
    teststrings = set()
    for i in range(base ** digits):
        entry = []
        for j in range(digits):
            i, m = divmod(i, base)
            entry.append(charset[m])
        teststrings.add(String("").join(entry))
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
    checkequal(9, String("abcdefghiabc"), "rfind", String("abc"))
    checkequal(12, String("abcdefghiabc"), "rfind", "")
    checkequal(0, String("abcdefghiabc"), "rfind", String("abcd"))
    checkequal(-1, String("abcdefghiabc"), "rfind", String("abcz"))

    checkequal(3, String("abc"), "rfind", String(""), 0)
    checkequal(3, String("abc"), "rfind", String(""), 3)
    checkequal(-1, String("abc"), "rfind", String(""), 4)

    # to check the ability to pass None as defaults
    checkequal(12, String("rrarrrrrrrrra"), "rfind", String("a"))
    checkequal(12, String("rrarrrrrrrrra"), "rfind", String("a"), 4)
    checkequal(-1, String("rrarrrrrrrrra"), "rfind", String("a"), 4, 6)
    checkequal(12, String("rrarrrrrrrrra"), "rfind", String("a"), 4, None)
    checkequal(2, String("rrarrrrrrrrra"), "rfind", String("a"), None, 6)

    checkraises(TypeError, String("hello"), "rfind")

    if contains_bytes:
        checkequal(-1, String("hello"), "rfind", 42)
    else:
        checkraises(TypeError, String("hello"), "rfind", 42)

    # For a variety of combinations,
    #    verify that str.rfind() matches __contains__
    #    and that the found substring is really at that location
    charset = [String(""), String("a"), String("b")]
    digits = 3
    base = len(charset)
    teststrings = set()
    for i in range(base ** digits):
        entry = []
        for j in range(digits):
            i, m = divmod(i, base)
            entry.append(charset[m])
        teststrings.add(String("").join(entry))
    for i in teststrings:
        for j in teststrings:
            loc = i.rfind(j)
            r1 = loc != -1
            r2 = j in i
            assert r1 == r2
            if loc != -1:
                assert i[loc : loc + len(j)] == j  # noqa: E203

    # issue 7458
    checkequal(-1, String("ab"), "rfind", String("xxx"), sys.maxsize + 1, 0)

    # issue #15534
    checkequal(0, String("<......\u043c..."), "rfind", "<")


def test_index():
    checkequal(0, String("abcdefghiabc"), "index", String(""))
    checkequal(3, String("abcdefghiabc"), "index", String("def"))
    checkequal(0, String("abcdefghiabc"), "index", String("abc"))
    checkequal(9, String("abcdefghiabc"), "index", "abc", 1)

    checkraises(ValueError, String("abcdefghiabc"), "index", String("hib"))
    checkraises(ValueError, String("abcdefghiab"), "index", String("abc"), 1)
    checkraises(ValueError, String("abcdefghi"), "index", String("ghi"), 8)
    checkraises(ValueError, String("abcdefghi"), "index", String("ghi"), -1)

    # to check the ability to pass None as defaults
    checkequal(2, String("rrarrrrrrrrra"), "index", String("a"))
    checkequal(12, String("rrarrrrrrrrra"), "index", String("a"), 4)
    checkraises(ValueError, String("rrarrrrrrrrra"), "index", String("a"), 4, 6)
    checkequal(12, String("rrarrrrrrrrra"), "index", String("a"), 4, None)
    checkequal(2, String("rrarrrrrrrrra"), "index", String("a"), None, 6)

    checkraises(TypeError, String("hello"), "index")

    checkraises(ValueError, String("hello"), "index", 42)


def test_rindex():
    checkequal(12, String("abcdefghiabc"), "rindex", String(""))
    checkequal(3, String("abcdefghiabc"), "rindex", String("def"))
    checkequal(9, String("abcdefghiabc"), "rindex", String("abc"))
    checkequal(0, String("abcdefghiabc"), "rindex", String("abc"), 0, -1)

    checkraises(ValueError, String("abcdefghiabc"), "rindex", String("hib"))
    checkraises(ValueError, String("defghiabc"), "rindex", String("def"), 1)
    checkraises(ValueError, String("defghiabc"), "rindex", "abc", 0, -1)
    checkraises(ValueError, String("abcdefghi"), "rindex", String("ghi"), 0, 8)
    checkraises(ValueError, String("abcdefghi"), "rindex", String("ghi"), 0, -1)

    # to check the ability to pass None as defaults
    checkequal(12, String("rrarrrrrrrrra"), "rindex", "a")
    checkequal(12, String("rrarrrrrrrrra"), "rindex", "a", 4)
    checkraises(ValueError, String("rrarrrrrrrrra"), "rindex", String("a"), 4, 6)
    checkequal(12, String("rrarrrrrrrrra"), "rindex", String("a"), 4, None)
    checkequal(2, String("rrarrrrrrrrra"), "rindex", String("a"), None, 6)

    checkraises(TypeError, "hello", "rindex")

    if contains_bytes:
        checkraises(ValueError, "hello", "rindex", 42)
    else:
        checkraises(TypeError, String("hello"), "rindex", 42)


def test_lower():
    checkequal(
        String("hello"),
        String("HeLLo"),
        "lower",
    )
    checkequal(String("hello"), String("hello"), "lower")
    checkraises(TypeError, String("hello"), "lower", 42)


def test_upper():
    checkequal(String("HELLO"), String("HeLLo"), "upper")
    checkequal(String("HELLO"), String("HELLO"), "upper")
    checkraises(TypeError, "hello", "upper", 42)


def test_expandtabs():
    checkequal(
        String("abc\rab      def\ng       hi"),
        String("abc\rab\tdef\ng\thi"),
        "expandtabs",
    )
    checkequal(
        String("abc\rab      def\ng       hi"),
        String("abc\rab\tdef\ng\thi"),
        "expandtabs",
        8,
    )
    checkequal("abc\rab  def\ng   hi", String("abc\rab\tdef\ng\thi"), "expandtabs", 4)
    checkequal(
        String("abc\r\nab      def\ng       hi"),
        String("abc\r\nab\tdef\ng\thi"),
        "expandtabs",
    )
    checkequal(
        String("abc\r\nab      def\ng       hi"),
        "abc\r\nab\tdef\ng\thi",
        "expandtabs",
        8,
    )
    checkequal(
        String("abc\r\nab  def\ng   hi"),
        "abc\r\nab\tdef\ng\thi",
        "expandtabs",
        4,
    )
    checkequal(
        String("abc\r\nab\r\ndef\ng\r\nhi"),
        String("abc\r\nab\r\ndef\ng\r\nhi"),
        "expandtabs",
        4,
    )
    # check keyword args
    checkequal(
        String("abc\rab      def\ng       hi"),
        String("abc\rab\tdef\ng\thi"),
        "expandtabs",
        tabsize=8,
    )
    checkequal(
        "abc\rab  def\ng   hi",
        String("abc\rab\tdef\ng\thi"),
        "expandtabs",
        tabsize=4,
    )

    checkequal("  a\n b", String(" \ta\n\tb"), "expandtabs", 1)

    checkraises(TypeError, String("hello"), "expandtabs", 42, 42)
    # This test is only valid when sizeof(int) == sizeof(void*) == 4.
    if sys.maxsize < (1 << 32) and struct.calcsize("P") == 4:
        checkraises(OverflowError, String("\ta\n\tb"), "expandtabs", sys.maxsize)


def test_split():
    # by a char
    checkequal(
        [String("a"), String("b"), "c", String("d")],
        String("a|b|c|d"),
        "split",
        "|",
    )
    checkequal(["a|b|c|d"], String("a|b|c|d"), "split", "|", 0)
    checkequal(
        ["a", String("b|c|d")],
        String("a|b|c|d"),
        "split",
        String("|"),
        1,
    )
    checkequal(["a", "b", "c|d"], String("a|b|c|d"), "split", "|", 2)
    checkequal(["a", "b", "c", "d"], String("a|b|c|d"), "split", String("|"), 3)
    checkequal(
        ["a", String("b"), String("c"), String("d")],
        String("a|b|c|d"),
        "split",
        "|",
        4,
    )
    checkequal(
        ["a", "b", "c", "d"],
        String("a|b|c|d"),
        "split",
        String("|"),
        sys.maxsize - 2,
    )
    checkequal(["a|b|c|d"], String("a|b|c|d"), "split", String("|"), 0)
    checkequal(
        [String("a"), String(""), String("b||c||d")],
        String("a||b||c||d"),
        "split",
        String("|"),
        2,
    )
    checkequal([String("abcd")], String("abcd"), "split", "|")
    checkequal([""], String(""), "split", "|")
    checkequal(
        [String("endcase "), String("")],
        String("endcase |"),
        "split",
        String("|"),
    )
    checkequal(
        ["", String(" startcase")],
        String("| startcase"),
        "split",
        String("|"),
    )
    checkequal(
        [String(""), String("bothcase"), String("")],
        String("|bothcase|"),
        "split",
        "|",
    )
    checkequal(
        ["a", "", String("b\x00c\x00d")],
        String("a\x00\x00b\x00c\x00d"),
        "split",
        "\x00",
        2,
    )

    checkequal(["a"] * 20, (String("a|") * 20)[:-1], "split", String("|"))
    checkequal(
        [String("a")] * 15 + [String("a|a|a|a|a")],
        ("a|" * 20)[:-1],
        "split",
        "|",
        15,
    )

    # by string
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a//b//c//d"),
        "split",
        String("//"),
    )
    checkequal(
        [String("a"), String("b//c//d")],
        String("a//b//c//d"),
        "split",
        "//",
        1,
    )
    checkequal(["a", "b", "c//d"], String("a//b//c//d"), "split", String("//"), 2)
    checkequal(["a", "b", "c", "d"], String("a//b//c//d"), "split", "//", 3)
    checkequal(
        ["a", "b", "c", String("d")],
        String("a//b//c//d"),
        "split",
        "//",
        4,
    )
    checkequal(
        ["a", "b", "c", "d"],
        String("a//b//c//d"),
        "split",
        String("//"),
        sys.maxsize - 10,
    )
    checkequal(
        [String("a//b//c//d")],
        String("a//b//c//d"),
        "split",
        String("//"),
        0,
    )
    checkequal(
        [String("a"), String(""), "b////c////d"],
        String("a////b////c////d"),
        "split",
        "//",
        2,
    )
    checkequal(
        [String(""), String(" bothcase "), String("")],
        String("test bothcase test"),
        "split",
        String("test"),
    )
    checkequal(
        [String("a"), String("bc")],
        String("abbbc"),
        "split",
        String("bb"),
    )
    checkequal(
        [String(""), String("")],
        String("aaa"),
        "split",
        String("aaa"),
    )
    checkequal([String("aaa")], String("aaa"), "split", String("aaa"), 0)
    checkequal(
        [String("ab"), String("ab")],
        String("abbaab"),
        "split",
        String("ba"),
    )
    checkequal([String("aaaa")], String("aaaa"), "split", String("aab"))
    checkequal([String("")], String(""), "split", String("aaa"))
    checkequal([String("aa")], String("aa"), "split", String("aaa"))
    checkequal(
        [String("A"), String("bobb")],
        String("Abbobbbobb"),
        "split",
        String("bbobb"),
    )
    checkequal(
        [String("A"), String("B"), String("")],
        String("AbbobbBbbobb"),
        "split",
        String("bbobb"),
    )

    checkequal(
        [String("a")] * 20,
        (String("aBLAH") * 20)[:-4],
        "split",
        String("BLAH"),
    )
    checkequal(
        [String("a")] * 20,
        (String("aBLAH") * 20)[:-4],
        "split",
        String("BLAH"),
        19,
    )
    checkequal(
        [String("a")] * 18 + [String("aBLAHa")],
        (String("aBLAH") * 20)[:-4],
        "split",
        String("BLAH"),
        18,
    )

    # with keyword args
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a|b|c|d"),
        "split",
        sep=String("|"),
    )
    checkequal(
        [String("a"), String("b|c|d")],
        String("a|b|c|d"),
        "split",
        String("|"),
        maxsplit=1,
    )
    checkequal(
        [String("a"), String("b|c|d")],
        String("a|b|c|d"),
        "split",
        sep=String("|"),
        maxsplit=1,
    )
    checkequal(
        [String("a"), String("b|c|d")],
        String("a|b|c|d"),
        "split",
        maxsplit=1,
        sep=String("|"),
    )
    checkequal(
        [String("a"), String("b c d")],
        String("a b c d"),
        "split",
        maxsplit=1,
    )

    # argument type
    checkraises(TypeError, String("hello"), "split", 42, 42, 42)

    # null case
    checkraises(ValueError, String("hello"), "split", String(""))
    checkraises(ValueError, String("hello"), "split", String(""), 0)


def test_rsplit():
    # by a char
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
    )
    checkequal(
        [String("a|b|c"), String("d")],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
        1,
    )
    checkequal(
        [String("a|b"), String("c"), String("d")],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
        2,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
        3,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
        4,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
        sys.maxsize - 100,
    )
    checkequal(
        [String("a|b|c|d")],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
        0,
    )
    checkequal(
        [String("a||b||c"), String(""), String("d")],
        String("a||b||c||d"),
        "rsplit",
        String("|"),
        2,
    )
    checkequal([String("abcd")], String("abcd"), "rsplit", String("|"))
    checkequal([String("")], String(""), "rsplit", String("|"))
    checkequal(
        [String(""), String(" begincase")],
        String("| begincase"),
        "rsplit",
        String("|"),
    )
    checkequal(
        [String("endcase "), String("")],
        String("endcase |"),
        "rsplit",
        String("|"),
    )
    checkequal(
        [String(""), String("bothcase"), String("")],
        String("|bothcase|"),
        "rsplit",
        String("|"),
    )

    checkequal(
        [String("a\x00\x00b"), String("c"), String("d")],
        String("a\x00\x00b\x00c\x00d"),
        "rsplit",
        String("\x00"),
        2,
    )

    checkequal(
        [String("a")] * 20,
        (String("a|") * 20)[:-1],
        "rsplit",
        String("|"),
    )
    checkequal(
        [String("a|a|a|a|a")] + [String("a")] * 15,
        (String("a|") * 20)[:-1],
        "rsplit",
        String("|"),
        15,
    )

    # by string
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a//b//c//d"),
        "rsplit",
        String("//"),
    )
    checkequal(
        [String("a//b//c"), String("d")],
        String("a//b//c//d"),
        "rsplit",
        String("//"),
        1,
    )
    checkequal(
        [String("a//b"), String("c"), String("d")],
        String("a//b//c//d"),
        "rsplit",
        String("//"),
        2,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a//b//c//d"),
        "rsplit",
        String("//"),
        3,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a//b//c//d"),
        "rsplit",
        String("//"),
        4,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a//b//c//d"),
        "rsplit",
        String("//"),
        sys.maxsize - 5,
    )
    checkequal(
        [String("a//b//c//d")],
        String("a//b//c//d"),
        "rsplit",
        String("//"),
        0,
    )
    checkequal(
        [String("a////b////c"), String(""), String("d")],
        String("a////b////c////d"),
        "rsplit",
        String("//"),
        2,
    )
    checkequal(
        [String(""), String(" begincase")],
        String("test begincase"),
        "rsplit",
        String("test"),
    )
    checkequal(
        [String("endcase "), String("")],
        String("endcase test"),
        "rsplit",
        String("test"),
    )
    checkequal(
        [String(""), String(" bothcase "), String("")],
        String("test bothcase test"),
        "rsplit",
        String("test"),
    )
    checkequal(
        [String("ab"), String("c")],
        String("abbbc"),
        "rsplit",
        String("bb"),
    )
    checkequal(
        [String(""), String("")],
        String("aaa"),
        "rsplit",
        String("aaa"),
    )
    checkequal([String("aaa")], String("aaa"), "rsplit", String("aaa"), 0)
    checkequal(
        [String("ab"), String("ab")],
        String("abbaab"),
        "rsplit",
        String("ba"),
    )
    checkequal([String("aaaa")], String("aaaa"), "rsplit", String("aab"))
    checkequal([String("")], String(""), "rsplit", String("aaa"))
    checkequal([String("aa")], String("aa"), "rsplit", String("aaa"))
    checkequal(
        [String("bbob"), String("A")],
        String("bbobbbobbA"),
        "rsplit",
        String("bbobb"),
    )
    checkequal(
        [String(""), String("B"), String("A")],
        String("bbobbBbbobbA"),
        "rsplit",
        String("bbobb"),
    )

    checkequal(
        [String("a")] * 20,
        (String("aBLAH") * 20)[:-4],
        "rsplit",
        String("BLAH"),
    )
    checkequal(
        [String("a")] * 20,
        (String("aBLAH") * 20)[:-4],
        "rsplit",
        String("BLAH"),
        19,
    )
    checkequal(
        [String("aBLAHa")] + [String("a")] * 18,
        (String("aBLAH") * 20)[:-4],
        "rsplit",
        String("BLAH"),
        18,
    )

    # with keyword args
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a|b|c|d"),
        "rsplit",
        sep=String("|"),
    )
    checkequal(
        [String("a|b|c"), String("d")],
        String("a|b|c|d"),
        "rsplit",
        String("|"),
        maxsplit=1,
    )
    checkequal(
        [String("a|b|c"), String("d")],
        String("a|b|c|d"),
        "rsplit",
        sep=String("|"),
        maxsplit=1,
    )
    checkequal(
        [String("a|b|c"), String("d")],
        String("a|b|c|d"),
        "rsplit",
        maxsplit=1,
        sep=String("|"),
    )
    checkequal(
        [String("a b c"), String("d")],
        String("a b c d"),
        "rsplit",
        maxsplit=1,
    )

    # argument type
    checkraises(TypeError, String("hello"), "rsplit", 42, 42, 42)

    # null case
    checkraises(ValueError, String("hello"), "rsplit", String(""))
    checkraises(ValueError, String("hello"), "rsplit", String(""), 0)


def test_replace():
    EQ = checkequal

    # Operations on the empty string
    EQ(
        String(""),
        String(""),
        "replace",
        String(""),
        String(""),
    )
    EQ(
        String("A"),
        String(""),
        "replace",
        String(""),
        String("A"),
    )
    EQ(
        String(""),
        String(""),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String(""),
        String(""),
        "replace",
        String("A"),
        String("A"),
    )
    EQ(
        String(""),
        String(""),
        "replace",
        String(""),
        String(""),
        100,
    )
    EQ(
        String(""),
        String(""),
        "replace",
        String(""),
        String(""),
        sys.maxsize,
    )

    # interleave (from==String(""), 'to' gets inserted everywhere)
    EQ(
        String("A"),
        String("A"),
        "replace",
        String(""),
        String(""),
    )
    EQ(
        String("*A*"),
        String("A"),
        "replace",
        String(""),
        String("*"),
    )
    EQ(
        String("*1A*1"),
        String("A"),
        "replace",
        String(""),
        String("*1"),
    )
    EQ(
        String("*-#A*-#"),
        String("A"),
        "replace",
        String(""),
        String("*-#"),
    )
    EQ(
        String("*-A*-A*-"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
    )
    EQ(
        String("*-A*-A*-"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
        -1,
    )
    EQ(
        String("*-A*-A*-"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
        sys.maxsize,
    )
    EQ(
        String("*-A*-A*-"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
        4,
    )
    EQ(
        String("*-A*-A*-"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
        3,
    )
    EQ(
        String("*-A*-A"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
        2,
    )
    EQ(
        String("*-AA"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
        1,
    )
    EQ(
        String("AA"),
        String("AA"),
        "replace",
        String(""),
        String("*-"),
        0,
    )

    # single character deletion (from==String("A"), to==String(""))
    EQ(
        String(""),
        String("A"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String(""),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String(""),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
        -1,
    )
    EQ(
        String(""),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
        sys.maxsize,
    )
    EQ(
        String(""),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
        4,
    )
    EQ(
        String(""),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
        3,
    )
    EQ(
        String("A"),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
        2,
    )
    EQ(
        String("AA"),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
        1,
    )
    EQ(
        String("AAA"),
        String("AAA"),
        "replace",
        String("A"),
        String(""),
        0,
    )
    EQ(
        String(""),
        String("AAAAAAAAAA"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String("BCD"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String("BCD"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        -1,
    )
    EQ(
        String("BCD"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        sys.maxsize,
    )
    EQ(
        String("BCD"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        5,
    )
    EQ(
        String("BCD"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        4,
    )
    EQ(
        String("BCDA"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        3,
    )
    EQ(
        String("BCADA"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        2,
    )
    EQ(
        String("BACADA"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        1,
    )
    EQ(
        String("ABACADA"),
        String("ABACADA"),
        "replace",
        String("A"),
        String(""),
        0,
    )
    EQ(
        String("BCD"),
        String("ABCAD"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String("BCD"),
        String("ABCADAA"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String("BCD"),
        String("BCD"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String("*************"),
        String("*************"),
        "replace",
        String("A"),
        String(""),
    )
    EQ(
        String("^A^"),
        String("^") + String("A") * 1000 + String("^"),
        "replace",
        String("A"),
        String(""),
        999,
    )

    # substring deletion (from==String("the"), to==String(""))
    EQ(
        String(""),
        String("the"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String("ater"),
        String("theater"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String(""),
        String("thethe"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String(""),
        String("thethethethe"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String("aaaa"),
        String("theatheatheathea"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String("that"),
        String("that"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String("thaet"),
        String("thaet"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String("here and re"),
        String("here and there"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String("here and re and re"),
        String("here and there and there"),
        "replace",
        String("the"),
        String(""),
        sys.maxsize,
    )
    EQ(
        String("here and re and re"),
        String("here and there and there"),
        "replace",
        String("the"),
        String(""),
        -1,
    )
    EQ(
        String("here and re and re"),
        String("here and there and there"),
        "replace",
        String("the"),
        String(""),
        3,
    )
    EQ(
        String("here and re and re"),
        String("here and there and there"),
        "replace",
        String("the"),
        String(""),
        2,
    )
    EQ(
        String("here and re and there"),
        String("here and there and there"),
        "replace",
        String("the"),
        String(""),
        1,
    )
    EQ(
        String("here and there and there"),
        String("here and there and there"),
        "replace",
        String("the"),
        String(""),
        0,
    )
    EQ(
        String("here and re and re"),
        String("here and there and there"),
        "replace",
        String("the"),
        String(""),
    )

    EQ(
        String("abc"),
        String("abc"),
        "replace",
        String("the"),
        String(""),
    )
    EQ(
        String("abcdefg"),
        String("abcdefg"),
        "replace",
        String("the"),
        String(""),
    )

    # substring deletion (from==String("bob"), to==String(""))
    EQ(
        String("bob"),
        String("bbobob"),
        "replace",
        String("bob"),
        String(""),
    )
    EQ(
        String("bobXbob"),
        String("bbobobXbbobob"),
        "replace",
        String("bob"),
        String(""),
    )
    EQ(
        String("aaaaaaa"),
        String("aaaaaaabob"),
        "replace",
        String("bob"),
        String(""),
    )
    EQ(
        String("aaaaaaa"),
        String("aaaaaaa"),
        "replace",
        String("bob"),
        String(""),
    )

    # single character replace in place (len(from)==len(to)==1)
    EQ(
        String("Who goes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("o"),
    )
    EQ(
        String("WhO gOes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("O"),
    )
    EQ(
        String("WhO gOes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("O"),
        sys.maxsize,
    )
    EQ(
        String("WhO gOes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("O"),
        -1,
    )
    EQ(
        String("WhO gOes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("O"),
        3,
    )
    EQ(
        String("WhO gOes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("O"),
        2,
    )
    EQ(
        String("WhO goes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("O"),
        1,
    )
    EQ(
        String("Who goes there?"),
        String("Who goes there?"),
        "replace",
        String("o"),
        String("O"),
        0,
    )

    EQ(
        String("Who goes there?"),
        String("Who goes there?"),
        "replace",
        String("a"),
        String("q"),
    )
    EQ(
        String("who goes there?"),
        String("Who goes there?"),
        "replace",
        String("W"),
        String("w"),
    )
    EQ(
        String("wwho goes there?ww"),
        String("WWho goes there?WW"),
        "replace",
        String("W"),
        String("w"),
    )
    EQ(
        String("Who goes there!"),
        String("Who goes there?"),
        "replace",
        String("?"),
        String("!"),
    )
    EQ(
        String("Who goes there!!"),
        String("Who goes there??"),
        "replace",
        String("?"),
        String("!"),
    )

    EQ(
        String("Who goes there?"),
        String("Who goes there?"),
        "replace",
        String("."),
        String("!"),
    )

    # substring replace in place (len(from)==len(to) > 1)
    EQ(
        String("Th** ** a t**sue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
    )
    EQ(
        String("Th** ** a t**sue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
        sys.maxsize,
    )
    EQ(
        String("Th** ** a t**sue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
        -1,
    )
    EQ(
        String("Th** ** a t**sue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
        4,
    )
    EQ(
        String("Th** ** a t**sue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
        3,
    )
    EQ(
        String("Th** ** a tissue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
        2,
    )
    EQ(
        String("Th** is a tissue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
        1,
    )
    EQ(
        String("This is a tissue"),
        String("This is a tissue"),
        "replace",
        String("is"),
        String("**"),
        0,
    )
    EQ(
        String("cobob"),
        String("bobob"),
        "replace",
        String("bob"),
        String("cob"),
    )
    EQ(
        String("cobobXcobocob"),
        String("bobobXbobobob"),
        "replace",
        String("bob"),
        String("cob"),
    )
    EQ(
        String("bobob"),
        String("bobob"),
        "replace",
        String("bot"),
        String("bot"),
    )

    # replace single character (len(from)==1, len(to)>1)
    EQ(
        String("ReyKKjaviKK"),
        String("Reykjavik"),
        "replace",
        String("k"),
        String("KK"),
    )
    EQ(
        String("ReyKKjaviKK"),
        String("Reykjavik"),
        "replace",
        String("k"),
        String("KK"),
        -1,
    )
    EQ(
        String("ReyKKjaviKK"),
        String("Reykjavik"),
        "replace",
        String("k"),
        String("KK"),
        sys.maxsize,
    )
    EQ(
        String("ReyKKjaviKK"),
        String("Reykjavik"),
        "replace",
        String("k"),
        String("KK"),
        2,
    )
    EQ(
        String("ReyKKjavik"),
        String("Reykjavik"),
        "replace",
        String("k"),
        String("KK"),
        1,
    )
    EQ(
        String("Reykjavik"),
        String("Reykjavik"),
        "replace",
        String("k"),
        String("KK"),
        0,
    )
    EQ(
        String("A----B----C----"),
        String("A.B.C."),
        "replace",
        String("."),
        String("----"),
    )
    # issue #15534
    EQ(
        String("...\u043c......&lt;"),
        String("...\u043c......<"),
        "replace",
        String("<"),
        String("&lt;"),
    )

    EQ(
        String("Reykjavik"),
        String("Reykjavik"),
        "replace",
        String("q"),
        String("KK"),
    )

    # replace substring (len(from)>1, len(to)!=len(from))
    EQ(
        String("ham, ham, eggs and ham"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
    )
    EQ(
        String("ham, ham, eggs and ham"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
        sys.maxsize,
    )
    EQ(
        String("ham, ham, eggs and ham"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
        -1,
    )
    EQ(
        String("ham, ham, eggs and ham"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
        4,
    )
    EQ(
        String("ham, ham, eggs and ham"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
        3,
    )
    EQ(
        String("ham, ham, eggs and spam"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
        2,
    )
    EQ(
        String("ham, spam, eggs and spam"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
        1,
    )
    EQ(
        String("spam, spam, eggs and spam"),
        String("spam, spam, eggs and spam"),
        "replace",
        String("spam"),
        String("ham"),
        0,
    )

    EQ(
        String("bobob"),
        String("bobobob"),
        "replace",
        String("bobob"),
        String("bob"),
    )
    EQ(
        String("bobobXbobob"),
        String("bobobobXbobobob"),
        "replace",
        String("bobob"),
        String("bob"),
    )
    EQ(
        String("BOBOBOB"),
        String("BOBOBOB"),
        "replace",
        String("bob"),
        String("bobby"),
    )

    checkequal(
        String("one@two!three!"),
        String("one!two!three!"),
        "replace",
        String("!"),
        String("@"),
        1,
    )
    checkequal(
        String("onetwothree"),
        String("one!two!three!"),
        "replace",
        String("!"),
        String(""),
    )
    checkequal(
        String("one@two@three!"),
        String("one!two!three!"),
        "replace",
        String("!"),
        String("@"),
        2,
    )
    checkequal(
        String("one@two@three@"),
        String("one!two!three!"),
        "replace",
        String("!"),
        String("@"),
        3,
    )
    checkequal(
        String("one@two@three@"),
        String("one!two!three!"),
        "replace",
        String("!"),
        String("@"),
        4,
    )
    checkequal(
        String("one!two!three!"),
        String("one!two!three!"),
        "replace",
        String("!"),
        String("@"),
        0,
    )
    checkequal(
        String("one@two@three@"),
        String("one!two!three!"),
        "replace",
        String("!"),
        String("@"),
    )
    checkequal(
        String("one!two!three!"),
        String("one!two!three!"),
        "replace",
        String("x"),
        String("@"),
    )
    checkequal(
        String("one!two!three!"),
        String("one!two!three!"),
        "replace",
        String("x"),
        String("@"),
        2,
    )
    checkequal(
        String("-a-b-c-"),
        String("abc"),
        "replace",
        String(""),
        String("-"),
    )
    checkequal(
        String("-a-b-c"),
        String("abc"),
        "replace",
        String(""),
        String("-"),
        3,
    )
    checkequal(
        String("abc"),
        String("abc"),
        "replace",
        String(""),
        String("-"),
        0,
    )
    checkequal(
        String(""),
        String(""),
        "replace",
        String(""),
        String(""),
    )
    checkequal(
        String("abc"),
        String("abc"),
        "replace",
        String("ab"),
        String("--"),
        0,
    )
    checkequal(
        String("abc"),
        String("abc"),
        "replace",
        String("xy"),
        String("--"),
    )
    # Next three for SF bug 422088: [OSF1 alpha] string.replace(); died with
    # MemoryError due to empty result (platform malloc issue when requesting
    # 0 bytes).
    checkequal(
        String(""),
        String("123"),
        "replace",
        String("123"),
        String(""),
    )
    checkequal(
        String(""),
        String("123123"),
        "replace",
        String("123"),
        String(""),
    )
    checkequal(
        String("x"),
        String("123x123"),
        "replace",
        String("123"),
        String(""),
    )

    checkraises(TypeError, String("hello"), "replace")
    checkraises(TypeError, String("hello"), "replace", 42)


if sys.version_info >= (3, 9):

    def test_removeprefix():
        checkequal("am", String("spam"), "removeprefix", String("sp"))
        checkequal("spamspam", String("spamspamspam"), "removeprefix", String("spam"))
        checkequal("spam", String("spam"), "removeprefix", String("python"))
        checkequal("spam", String("spam"), "removeprefix", String("spider"))
        checkequal("spam", String("spam"), "removeprefix", String("spam and eggs"))

        checkequal("", String(""), "removeprefix", String(""))
        checkequal("", String(""), "removeprefix", String("abcde"))
        checkequal("abcde", String("abcde"), "removeprefix", String(""))
        checkequal("", String("abcde"), "removeprefix", String("abcde"))

        checkraises(TypeError, String("hello"), "removeprefix")
        checkraises(TypeError, String("hello"), "removeprefix", 42)
        checkraises(TypeError, String("hello"), "removeprefix", 42, "h")
        checkraises(TypeError, String("hello"), "removeprefix", "h", 42)
        checkraises(TypeError, String("hello"), "removeprefix", ("he", "l"))

    def test_removesuffix():
        checkequal("sp", String("spam"), "removesuffix", String("am"))
        checkequal("spamspam", String("spamspamspam"), "removesuffix", String("spam"))
        checkequal("spam", String("spam"), "removesuffix", String("python"))
        checkequal("spam", String("spam"), "removesuffix", String("blam"))
        checkequal("spam", String("spam"), "removesuffix", String("eggs and spam"))

        checkequal("", String(""), "removesuffix", String(""))
        checkequal("", String(""), "removesuffix", String("abcde"))
        checkequal("abcde", String("abcde"), "removesuffix", String(""))
        checkequal("", String("abcde"), "removesuffix", String("abcde"))

        checkraises(TypeError, String("hello"), "removesuffix")
        checkraises(TypeError, String("hello"), "removesuffix", 42)
        checkraises(TypeError, String("hello"), "removesuffix", 42, "h")
        checkraises(TypeError, String("hello"), "removesuffix", "h", 42)
        checkraises(TypeError, String("hello"), "removesuffix", ("lo", "l"))


@pytest.mark.xfail
def test_multiple_arguments():
    checkraises(TypeError, String("hello"), "replace", 42, String("h"))
    checkraises(TypeError, String("hello"), "replace", String("h"), 42)


def test_capitalize():
    checkequal(String(" hello "), String(" hello "), "capitalize")
    checkequal(String("Hello "), String("Hello "), "capitalize")
    checkequal(String("Hello "), String("hello "), "capitalize")
    checkequal(String("Aaaa"), String("aaaa"), "capitalize")
    checkequal(String("Aaaa"), String("AaAa"), "capitalize")

    checkraises(TypeError, String("hello"), "capitalize", 42)


def test_additional_split():
    checkequal(
        [
            String("this"),
            String("is"),
            String("the"),
            "split",
            String("function"),
        ],
        String("this is the split function"),
        "split",
    )

    # by whitespace
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d "),
        "split",
    )
    checkequal(
        [String("a"), String("b c d")],
        String("a b c d"),
        "split",
        None,
        1,
    )
    checkequal(
        [String("a"), String("b"), String("c d")],
        String("a b c d"),
        "split",
        None,
        2,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d"),
        "split",
        None,
        3,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d"),
        "split",
        None,
        4,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d"),
        "split",
        None,
        sys.maxsize - 1,
    )
    checkequal([String("a b c d")], String("a b c d"), "split", None, 0)
    checkequal([String("a b c d")], String("  a b c d"), "split", None, 0)
    checkequal(
        [String("a"), String("b"), String("c  d")],
        String("a  b  c  d"),
        "split",
        None,
        2,
    )

    checkequal([], String("         "), "split")
    checkequal([String("a")], String("  a    "), "split")
    checkequal([String("a"), String("b")], String("  a    b   "), "split")
    checkequal(
        [String("a"), String("b   ")],
        String("  a    b   "),
        "split",
        None,
        1,
    )
    checkequal(
        [String("a    b   c   ")],
        String("  a    b   c   "),
        "split",
        None,
        0,
    )
    checkequal(
        [String("a"), String("b   c   ")],
        String("  a    b   c   "),
        "split",
        None,
        1,
    )
    checkequal(
        [String("a"), String("b"), String("c   ")],
        String("  a    b   c   "),
        "split",
        None,
        2,
    )
    checkequal(
        [String("a"), String("b"), String("c")],
        String("  a    b   c   "),
        "split",
        None,
        3,
    )
    checkequal(
        [String("a"), String("b")],
        String("\n\ta \t\r b \v "),
        "split",
    )
    aaa = String(" a ") * 20
    checkequal([String("a")] * 20, aaa, "split")
    checkequal([String("a")] + [aaa[4:]], aaa, "split", None, 1)
    checkequal([String("a")] * 19 + [String("a ")], aaa, "split", None, 19)

    for b in (
        String("arf\tbarf"),
        String("arf\nbarf"),
        String("arf\rbarf"),
        String("arf\fbarf"),
        String("arf\vbarf"),
    ):
        checkequal([String("arf"), String("barf")], b, "split")
        checkequal([String("arf"), String("barf")], b, "split", None)
        checkequal([String("arf"), String("barf")], b, "split", None, 2)


def test_additional_rsplit():
    checkequal(
        [
            String("this"),
            String("is"),
            String("the"),
            "rsplit",
            String("function"),
        ],
        String("this is the rsplit function"),
        "rsplit",
    )

    # by whitespace
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d "),
        "rsplit",
    )
    checkequal(
        [String("a b c"), String("d")],
        String("a b c d"),
        "rsplit",
        None,
        1,
    )
    checkequal(
        [String("a b"), String("c"), String("d")],
        String("a b c d"),
        "rsplit",
        None,
        2,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d"),
        "rsplit",
        None,
        3,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d"),
        "rsplit",
        None,
        4,
    )
    checkequal(
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
        String("a b c d"),
        "rsplit",
        None,
        sys.maxsize - 20,
    )
    checkequal([String("a b c d")], String("a b c d"), "rsplit", None, 0)
    checkequal([String("a b c d")], String("a b c d  "), "rsplit", None, 0)
    checkequal(
        [String("a  b"), String("c"), String("d")],
        String("a  b  c  d"),
        "rsplit",
        None,
        2,
    )

    checkequal([], String("         "), "rsplit")
    checkequal([String("a")], String("  a    "), "rsplit")
    checkequal([String("a"), String("b")], String("  a    b   "), "rsplit")
    checkequal(
        [String("  a"), String("b")],
        String("  a    b   "),
        "rsplit",
        None,
        1,
    )
    checkequal(
        [String("  a    b   c")],
        String("  a    b   c   "),
        "rsplit",
        None,
        0,
    )
    checkequal(
        [String("  a    b"), String("c")],
        String("  a    b   c   "),
        "rsplit",
        None,
        1,
    )
    checkequal(
        [String("  a"), String("b"), String("c")],
        String("  a    b   c   "),
        "rsplit",
        None,
        2,
    )
    checkequal(
        [String("a"), String("b"), String("c")],
        String("  a    b   c   "),
        "rsplit",
        None,
        3,
    )
    checkequal(
        [String("a"), String("b")],
        String("\n\ta \t\r b \v "),
        "rsplit",
        None,
        88,
    )
    aaa = String(" a ") * 20
    checkequal([String("a")] * 20, aaa, "rsplit")
    checkequal([aaa[:-4]] + [String("a")], aaa, "rsplit", None, 1)
    checkequal([String(" a  a")] + [String("a")] * 18, aaa, "rsplit", None, 18)

    for b in (
        String("arf\tbarf"),
        String("arf\nbarf"),
        String("arf\rbarf"),
        String("arf\fbarf"),
        String("arf\vbarf"),
    ):
        checkequal([String("arf"), String("barf")], b, "rsplit")
        checkequal([String("arf"), String("barf")], b, "rsplit", None)
        checkequal([String("arf"), String("barf")], b, "rsplit", None, 2)


def test_strip_whitespace():
    checkequal(String("hello"), String("   hello   "), "strip")
    checkequal(String("hello   "), String("   hello   "), "lstrip")
    checkequal(String("   hello"), String("   hello   "), "rstrip")
    checkequal(String("hello"), String("hello"), "strip")

    b = String(" \t\n\r\f\vabc \t\n\r\f\v")
    checkequal(String("abc"), b, "strip")
    checkequal(String("abc \t\n\r\f\v"), b, "lstrip")
    checkequal(String(" \t\n\r\f\vabc"), b, "rstrip")

    # strip/lstrip/rstrip with None arg
    checkequal(String("hello"), String("   hello   "), "strip", None)
    checkequal(String("hello   "), String("   hello   "), "lstrip", None)
    checkequal(String("   hello"), String("   hello   "), "rstrip", None)
    checkequal(String("hello"), String("hello"), "strip", None)


def test_strip():
    # strip/lstrip/rstrip with str arg
    checkequal(
        String("hello"),
        String("xyzzyhelloxyzzy"),
        "strip",
        String("xyz"),
    )
    checkequal(
        String("helloxyzzy"),
        String("xyzzyhelloxyzzy"),
        "lstrip",
        String("xyz"),
    )
    checkequal(
        String("xyzzyhello"),
        String("xyzzyhelloxyzzy"),
        "rstrip",
        String("xyz"),
    )
    checkequal(String("hello"), String("hello"), "strip", String("xyz"))
    checkequal(
        String(""),
        String("mississippi"),
        "strip",
        String("mississippi"),
    )

    # only trim the start and end; does not strip internal characters
    checkequal(
        String("mississipp"),
        String("mississippi"),
        "strip",
        String("i"),
    )

    checkraises(TypeError, String("hello"), "strip", 42, 42)
    checkraises(TypeError, String("hello"), "lstrip", 42, 42)
    checkraises(TypeError, String("hello"), "rstrip", 42, 42)


def test_ljust():
    checkequal(String("abc       "), String("abc"), "ljust", 10)
    checkequal(String("abc   "), String("abc"), "ljust", 6)
    checkequal(String("abc"), String("abc"), "ljust", 3)
    checkequal(String("abc"), String("abc"), "ljust", 2)
    checkequal(
        String("abc*******"),
        String("abc"),
        "ljust",
        10,
        String("*"),
    )
    checkraises(TypeError, String("abc"), "ljust")


def test_rjust():
    checkequal(String("       abc"), String("abc"), "rjust", 10)
    checkequal(String("   abc"), String("abc"), "rjust", 6)
    checkequal(String("abc"), String("abc"), "rjust", 3)
    checkequal(String("abc"), String("abc"), "rjust", 2)
    checkequal(
        String("*******abc"),
        String("abc"),
        "rjust",
        10,
        String("*"),
    )
    checkraises(TypeError, String("abc"), "rjust")


def test_center():
    checkequal(String("   abc    "), String("abc"), "center", 10)
    checkequal(String(" abc  "), String("abc"), "center", 6)
    checkequal(String("abc"), String("abc"), "center", 3)
    checkequal(String("abc"), String("abc"), "center", 2)
    checkequal(
        String("***abc****"),
        String("abc"),
        "center",
        10,
        String("*"),
    )
    checkraises(TypeError, String("abc"), "center")


def test_swapcase():
    checkequal(String("hEllO CoMPuTErS"), String("HeLLo cOmpUteRs"), "swapcase")
    checkraises(TypeError, String("hello"), "swapcase", 42)


def test_zfill():
    checkequal(String("123"), String("123"), "zfill", 2)
    checkequal(String("123"), String("123"), "zfill", 3)
    checkequal(String("0123"), String("123"), "zfill", 4)
    checkequal(String("+123"), String("+123"), "zfill", 3)
    checkequal(String("+123"), String("+123"), "zfill", 4)
    checkequal(String("+0123"), String("+123"), "zfill", 5)
    checkequal(String("-123"), String("-123"), "zfill", 3)
    checkequal(String("-123"), String("-123"), "zfill", 4)
    checkequal(String("-0123"), String("-123"), "zfill", 5)
    checkequal(String("000"), String(""), "zfill", 3)
    checkequal(String("34"), String("34"), "zfill", 1)
    checkequal(String("0034"), String("34"), "zfill", 4)

    checkraises(TypeError, String("123"), "zfill")


def test_islower():
    checkequal(False, String(""), "islower")
    checkequal(True, String("a"), "islower")
    checkequal(False, String("A"), "islower")
    checkequal(False, String("\n"), "islower")
    checkequal(True, String("abc"), "islower")
    checkequal(False, String("aBc"), "islower")
    checkequal(True, String("abc\n"), "islower")
    checkraises(TypeError, String("abc"), "islower", 42)


def test_isupper():
    checkequal(False, String(""), "isupper")
    checkequal(False, String("a"), "isupper")
    checkequal(True, String("A"), "isupper")
    checkequal(False, String("\n"), "isupper")
    checkequal(True, String("ABC"), "isupper")
    checkequal(False, String("AbC"), "isupper")
    checkequal(True, String("ABC\n"), "isupper")
    checkraises(TypeError, String("abc"), "isupper", 42)


def test_istitle():
    checkequal(False, String(""), "istitle")
    checkequal(False, String("a"), "istitle")
    checkequal(True, String("A"), "istitle")
    checkequal(False, String("\n"), "istitle")
    checkequal(True, String("A Titlecased Line"), "istitle")
    checkequal(True, String("A\nTitlecased Line"), "istitle")
    checkequal(True, String("A Titlecased, Line"), "istitle")
    checkequal(False, String("Not a capitalized String"), "istitle")
    checkequal(False, String("Not\ta Titlecase String"), "istitle")
    checkequal(False, String("Not--a Titlecase String"), "istitle")
    checkequal(False, String("NOT"), "istitle")
    checkraises(TypeError, String("abc"), "istitle", 42)


def test_isspace():
    checkequal(False, String(""), "isspace")
    checkequal(False, String("a"), "isspace")
    checkequal(True, String(" "), "isspace")
    checkequal(True, String("\t"), "isspace")
    checkequal(True, String("\r"), "isspace")
    checkequal(True, String("\n"), "isspace")
    checkequal(True, String(" \t\r\n"), "isspace")
    checkequal(False, String(" \t\r\na"), "isspace")
    checkraises(TypeError, String("abc"), "isspace", 42)


def test_isalpha():
    checkequal(False, String(""), "isalpha")
    checkequal(True, String("a"), "isalpha")
    checkequal(True, String("A"), "isalpha")
    checkequal(False, String("\n"), "isalpha")
    checkequal(True, String("abc"), "isalpha")
    checkequal(False, String("aBc123"), "isalpha")
    checkequal(False, String("abc\n"), "isalpha")
    checkraises(TypeError, String("abc"), "isalpha", 42)


def test_isalnum():
    checkequal(False, String(""), "isalnum")
    checkequal(True, String("a"), "isalnum")
    checkequal(True, String("A"), "isalnum")
    checkequal(False, String("\n"), "isalnum")
    checkequal(True, String("123abc456"), "isalnum")
    checkequal(True, String("a1b3c"), "isalnum")
    checkequal(False, String("aBc000 "), "isalnum")
    checkequal(False, String("abc\n"), "isalnum")
    checkraises(TypeError, String("abc"), "isalnum", 42)


def test_isascii():
    if sys.version_info >= (3, 7):
        checkequal(True, String(""), "isascii")
        checkequal(True, String("\x00"), "isascii")
        checkequal(True, String("\x7f"), "isascii")
        checkequal(True, String("\x00\x7f"), "isascii")
        checkequal(False, String("\x80"), "isascii")
        checkequal(False, String("\xe9"), "isascii")
        # bytes.isascii() and bytearray.isascii() has optimization which
        # check 4 or 8 bytes at once.  So check some alignments.
        for p in range(8):
            checkequal(True, String(" ") * p + String("\x7f"), "isascii")
            checkequal(False, String(" ") * p + String("\x80"), "isascii")
            checkequal(
                True,
                String(" ") * p + String("\x7f") + String(" ") * 8,
                "isascii",
            )
            checkequal(
                False,
                String(" ") * p + String("\x80") + String(" ") * 8,
                "isascii",
            )
    else:
        with pytest.raises(AttributeError):
            checkequal(True, String(""), "isascii")


def test_isdigit():
    checkequal(False, String(""), "isdigit")
    checkequal(False, String("a"), "isdigit")
    checkequal(True, String("0"), "isdigit")
    checkequal(True, String("0123456789"), "isdigit")
    checkequal(False, String("0123456789a"), "isdigit")

    checkraises(TypeError, String("abc"), "isdigit", 42)


def test_title():
    checkequal(String(" Hello "), String(" hello "), "title")
    checkequal(String("Hello "), String("hello "), "title")
    checkequal(String("Hello "), String("Hello "), "title")
    checkequal(
        String("Format This As Title String"),
        String("fOrMaT thIs aS titLe String"),
        "title",
    )
    checkequal(
        String("Format,This-As*Title;String"),
        String("fOrMaT,thIs-aS*titLe;String"),
        "title",
    )
    checkequal(String("Getint"), String("getInt"), "title")
    checkraises(TypeError, String("hello"), "title", 42)


def test_splitlines():
    checkequal(
        [
            String("abc"),
            String("def"),
            String(""),
            String("ghi"),
        ],
        String("abc\ndef\n\rghi"),
        "splitlines",
    )
    checkequal(
        [
            String("abc"),
            String("def"),
            String(""),
            String("ghi"),
        ],
        String("abc\ndef\n\r\nghi"),
        "splitlines",
    )
    checkequal(
        [String("abc"), String("def"), String("ghi")],
        String("abc\ndef\r\nghi"),
        "splitlines",
    )
    checkequal(
        [String("abc"), String("def"), String("ghi")],
        String("abc\ndef\r\nghi\n"),
        "splitlines",
    )
    checkequal(
        [
            String("abc"),
            String("def"),
            String("ghi"),
            String(""),
        ],
        String("abc\ndef\r\nghi\n\r"),
        "splitlines",
    )
    checkequal(
        [
            String(""),
            String("abc"),
            String("def"),
            String("ghi"),
            String(""),
        ],
        String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
    )
    checkequal(
        [
            String(""),
            String("abc"),
            String("def"),
            String("ghi"),
            String(""),
        ],
        String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        False,
    )
    checkequal(
        [
            String("\n"),
            String("abc\n"),
            String("def\r\n"),
            String("ghi\n"),
            String("\r"),
        ],
        String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        True,
    )
    checkequal(
        [
            String(""),
            String("abc"),
            String("def"),
            String("ghi"),
            String(""),
        ],
        String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        keepends=False,
    )
    checkequal(
        [
            String("\n"),
            String("abc\n"),
            String("def\r\n"),
            String("ghi\n"),
            String("\r"),
        ],
        String("\nabc\ndef\r\nghi\n\r"),
        "splitlines",
        keepends=True,
    )

    checkraises(TypeError, String("abc"), "splitlines", 42, 42)


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
            String("\u1ffc\u1ff3\u1ff3\u1ff3"),
            String("\u1ff3\u1ff3\u1ffc\u1ffc"),
            "capitalize",
        )
    else:
        # a, b, capitalize
        # ΩΙῳῳῳ, ῳῳῼῼ, capitalize
        checkequal(
            String("\u03a9\u0399\u1ff3\u1ff3\u1ff3"),
            String("\u1ff3\u1ff3\u1ffc\u1ffc"),
            "capitalize",
        )
    # check with cased non-letter chars
    checkequal(
        String("\u24c5\u24e8\u24e3\u24d7\u24de\u24dd"),
        String("\u24c5\u24ce\u24c9\u24bd\u24c4\u24c3"),
        "capitalize",
    )
    checkequal(
        String("\u24c5\u24e8\u24e3\u24d7\u24de\u24dd"),
        String("\u24df\u24e8\u24e3\u24d7\u24de\u24dd"),
        "capitalize",
    )
    checkequal(
        String("\u2160\u2171\u2172"),
        String("\u2160\u2161\u2162"),
        "capitalize",
    )
    checkequal(
        String("\u2160\u2171\u2172"),
        String("\u2170\u2171\u2172"),
        "capitalize",
    )
    # check with Ll chars with no upper - nothing changes here
    checkequal(
        String("\u019b\u1d00\u1d86\u0221\u1fb7"),
        String("\u019b\u1d00\u1d86\u0221\u1fb7"),
        "capitalize",
    )


def test_startswith():
    checkequal(True, String("hello"), "startswith", String("he"))
    checkequal(True, String("hello"), "startswith", String("hello"))
    checkequal(False, String("hello"), "startswith", String("hello world"))
    checkequal(True, String("hello"), "startswith", String(""))
    checkequal(False, String("hello"), "startswith", String("ello"))
    checkequal(True, String("hello"), "startswith", String("ello"), 1)
    checkequal(True, String("hello"), "startswith", String("o"), 4)
    checkequal(False, String("hello"), "startswith", String("o"), 5)
    checkequal(True, String("hello"), "startswith", String(""), 5)
    checkequal(False, String("hello"), "startswith", String("lo"), 6)
    checkequal(True, String("helloworld"), "startswith", String("lowo"), 3)
    checkequal(True, String("helloworld"), "startswith", String("lowo"), 3, 7)
    checkequal(False, String("helloworld"), "startswith", String("lowo"), 3, 6)
    checkequal(True, String(""), "startswith", String(""), 0, 1)
    checkequal(True, String(""), "startswith", String(""), 0, 0)
    checkequal(False, String(""), "startswith", String(""), 1, 0)

    # test negative indices
    checkequal(True, String("hello"), "startswith", String("he"), 0, -1)
    checkequal(True, String("hello"), "startswith", String("he"), -53, -1)
    checkequal(False, String("hello"), "startswith", String("hello"), 0, -1)
    checkequal(
        False,
        String("hello"),
        "startswith",
        String("hello world"),
        -1,
        -10,
    )
    checkequal(False, String("hello"), "startswith", String("ello"), -5)
    checkequal(True, String("hello"), "startswith", String("ello"), -4)
    checkequal(False, String("hello"), "startswith", String("o"), -2)
    checkequal(True, String("hello"), "startswith", String("o"), -1)
    checkequal(True, String("hello"), "startswith", String(""), -3, -3)
    checkequal(False, String("hello"), "startswith", String("lo"), -9)

    checkraises(TypeError, String("hello"), "startswith")
    checkraises(TypeError, String("hello"), "startswith", 42)

    # test tuple arguments
    checkequal(
        True,
        String("hello"),
        "startswith",
        (String("he"), String("ha")),
    )
    checkequal(
        False,
        String("hello"),
        "startswith",
        (String("lo"), String("llo")),
    )
    checkequal(
        True,
        String("hello"),
        "startswith",
        (String("hellox"), String("hello")),
    )
    checkequal(False, String("hello"), "startswith", ())
    checkequal(
        True,
        String("helloworld"),
        "startswith",
        (String("hellowo"), String("rld"), String("lowo")),
        3,
    )
    checkequal(
        False,
        String("helloworld"),
        "startswith",
        (String("hellowo"), String("ello"), String("rld")),
        3,
    )
    checkequal(
        True,
        String("hello"),
        "startswith",
        (String("lo"), String("he")),
        0,
        -1,
    )
    checkequal(
        False,
        String("hello"),
        "startswith",
        (String("he"), String("hel")),
        0,
        1,
    )
    checkequal(
        True,
        String("hello"),
        "startswith",
        (String("he"), String("hel")),
        0,
        2,
    )

    checkraises(TypeError, String("hello"), "startswith", (42,))


def test_endswith():
    checkequal(True, String("hello"), "endswith", String("lo"))
    checkequal(False, String("hello"), "endswith", String("he"))
    checkequal(True, String("hello"), "endswith", String(""))
    checkequal(False, String("hello"), "endswith", String("hello world"))
    checkequal(False, String("helloworld"), "endswith", String("worl"))
    checkequal(True, String("helloworld"), "endswith", String("worl"), 3, 9)
    checkequal(True, String("helloworld"), "endswith", String("world"), 3, 12)
    checkequal(True, String("helloworld"), "endswith", String("lowo"), 1, 7)
    checkequal(True, String("helloworld"), "endswith", String("lowo"), 2, 7)
    checkequal(True, String("helloworld"), "endswith", String("lowo"), 3, 7)
    checkequal(False, String("helloworld"), "endswith", String("lowo"), 4, 7)
    checkequal(False, String("helloworld"), "endswith", String("lowo"), 3, 8)
    checkequal(False, String("ab"), "endswith", String("ab"), 0, 1)
    checkequal(False, String("ab"), "endswith", String("ab"), 0, 0)
    checkequal(True, String(""), "endswith", String(""), 0, 1)
    checkequal(True, String(""), "endswith", String(""), 0, 0)
    checkequal(False, String(""), "endswith", String(""), 1, 0)

    # test negative indices
    checkequal(True, String("hello"), "endswith", String("lo"), -2)
    checkequal(False, String("hello"), "endswith", String("he"), -2)
    checkequal(True, String("hello"), "endswith", String(""), -3, -3)
    checkequal(False, String("hello"), "endswith", String("hello world"), -10, -2)
    checkequal(False, String("helloworld"), "endswith", String("worl"), -6)
    checkequal(True, String("helloworld"), "endswith", String("worl"), -5, -1)
    checkequal(True, String("helloworld"), "endswith", String("worl"), -5, 9)
    checkequal(True, String("helloworld"), "endswith", String("world"), -7, 12)
    checkequal(True, String("helloworld"), "endswith", String("lowo"), -99, -3)
    checkequal(True, String("helloworld"), "endswith", String("lowo"), -8, -3)
    checkequal(True, String("helloworld"), "endswith", String("lowo"), -7, -3)
    checkequal(False, String("helloworld"), "endswith", String("lowo"), 3, -4)
    checkequal(False, String("helloworld"), "endswith", String("lowo"), -8, -2)

    checkraises(TypeError, String("hello"), "endswith")
    checkraises(TypeError, String("hello"), "endswith", 42)

    # test tuple arguments
    checkequal(
        False,
        String("hello"),
        "endswith",
        (String("he"), String("ha")),
    )
    checkequal(
        True,
        String("hello"),
        "endswith",
        (String("lo"), String("llo")),
    )
    checkequal(
        True,
        String("hello"),
        "endswith",
        (String("hellox"), String("hello")),
    )
    checkequal(False, String("hello"), "endswith", ())
    checkequal(
        True,
        String("helloworld"),
        "endswith",
        (String("hellowo"), String("rld"), String("lowo")),
        3,
    )
    checkequal(
        False,
        String("helloworld"),
        "endswith",
        (String("hellowo"), String("ello"), String("rld")),
        3,
        -1,
    )
    checkequal(
        True,
        String("hello"),
        "endswith",
        (String("hell"), String("ell")),
        0,
        -1,
    )
    checkequal(
        False,
        String("hello"),
        "endswith",
        (String("he"), String("hel")),
        0,
        1,
    )
    checkequal(
        True,
        String("hello"),
        "endswith",
        (String("he"), String("hell")),
        0,
        4,
    )

    checkraises(TypeError, String("hello"), "endswith", (42,))


def test___contains__():
    checkequal(True, String(""), "__contains__", String(""))
    checkequal(True, String("abc"), "__contains__", String(""))
    checkequal(False, String("abc"), "__contains__", String("\0"))
    checkequal(True, String("\0abc"), "__contains__", String("\0"))
    checkequal(True, String("abc\0"), "__contains__", String("\0"))
    checkequal(True, String("\0abc"), "__contains__", String("a"))
    checkequal(True, String("asdf"), "__contains__", String("asdf"))
    checkequal(False, String("asd"), "__contains__", String("asdf"))
    checkequal(False, String(""), "__contains__", String("asdf"))


def test_subscript():
    checkequal(String("a"), String("abc"), "__getitem__", 0)
    checkequal(String("c"), String("abc"), "__getitem__", -1)
    checkequal(String("a"), String("abc"), "__getitem__", 0)
    checkequal(String("abc"), String("abc"), "__getitem__", slice(0, 3))
    checkequal(String("abc"), String("abc"), "__getitem__", slice(0, 1000))
    checkequal(String("a"), String("abc"), "__getitem__", slice(0, 1))
    checkequal(String(""), String("abc"), "__getitem__", slice(0, 0))

    checkraises(TypeError, String("abc"), "__getitem__", String("def"))


def test_slice():
    checkequal(String("abc"), String("abc"), "__getitem__", slice(0, 1000))
    checkequal(String("abc"), String("abc"), "__getitem__", slice(0, 3))
    checkequal(String("ab"), String("abc"), "__getitem__", slice(0, 2))
    checkequal(String("bc"), String("abc"), "__getitem__", slice(1, 3))
    checkequal(String("b"), String("abc"), "__getitem__", slice(1, 2))
    checkequal(String(""), String("abc"), "__getitem__", slice(2, 2))
    checkequal(String(""), String("abc"), "__getitem__", slice(1000, 1000))
    checkequal(String(""), String("abc"), "__getitem__", slice(2000, 1000))
    checkequal(String(""), String("abc"), "__getitem__", slice(2, 1))

    checkraises(TypeError, String("abc"), "__getitem__", String("def"))


@pytest.mark.slow
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
                    String("").join(L),
                    s,
                    "__getitem__",
                    slice(start, stop, step),
                )


def test_mul():
    checkequal(String(""), String("abc"), "__mul__", -1)
    checkequal(String(""), String("abc"), "__mul__", 0)
    checkequal(String("abc"), String("abc"), "__mul__", 1)
    checkequal(String("abcabcabc"), String("abc"), "__mul__", 3)
    checkraises(TypeError, String("abc"), "__mul__")
    checkraises(TypeError, String("abc"), "__mul__", String(""))
    # XXX: on a 64-bit system, this doesn't raise an overflow error,
    # but either raises a MemoryError, or succeeds (if you have 54TiB)
    # checkraises(OverflowError, 10000*'abc', '__mul__', 2000000000)


@pytest.mark.slow
def test_join():
    checkequal(
        String("a b c d"),
        String(" "),
        "join",
        [
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ],
    )
    checkequal(
        String("abcd"),
        String(""),
        "join",
        (
            String("a"),
            String("b"),
            String("c"),
            String("d"),
        ),
    )
    checkequal(
        String("bd"),
        String(""),
        "join",
        (String(""), String("b"), String(""), String("d")),
    )
    checkequal(
        String("ac"),
        String(""),
        "join",
        (String("a"), String(""), String("c"), String("")),
    )
    checkequal(String("w x y z"), String(" "), "join", Sequence())
    checkequal(String("abc"), String("a"), "join", (String("abc"),))
    checkequal(
        String("z"),
        String("a"),
        "join",
        List([String("z")]),
    )
    checkequal(
        String("a.b.c"),
        String("."),
        "join",
        [String("a"), String("b"), String("c")],
    )
    checkraises(
        TypeError,
        String("."),
        "join",
        [String("a"), String("b"), 3],
    )
    for i in [5, 25, 125]:
        checkequal(
            (((String("a") * i) + String("-")) * i)[:-1],
            String("-"),
            "join",
            [String("a") * i] * i,
        )
        checkequal(
            (((String("a") * i) + String("-")) * i)[:-1],
            String("-"),
            "join",
            (String("a") * i,) * i,
        )

    checkequal(String("a b c"), String(" "), "join", BadSeq2())

    checkraises(TypeError, String(" "), "join")
    checkraises(TypeError, String(" "), "join", None)
    checkraises(TypeError, String(" "), "join", 7)
    checkraises(TypeError, String(" "), "join", [1, 2, bytes()])


def test_formatting():
    checkequal(
        String("+hello+"),
        String("+%s+"),
        "__mod__",
        String("hello"),
    )
    checkequal(String("+10+"), String("+%d+"), "__mod__", 10)
    checkequal(String("a"), String("%c"), "__mod__", String("a"))
    checkequal(String("a"), String("%c"), "__mod__", String("a"))
    checkequal(String("$"), String("%c"), "__mod__", 36)
    checkequal(String("10"), String("%d"), "__mod__", 10)
    checkequal(String("\x7f"), String("%c"), "__mod__", 0x7F)

    for ordinal in (-100, 0x200000):
        # unicode raises ValueError, str raises OverflowError
        checkraises((ValueError, OverflowError), String("%c"), "__mod__", ordinal)

    longvalue = sys.maxsize + 10
    slongvalue = str(longvalue)
    checkequal(String(" 42"), String("%3ld"), "__mod__", 42)
    checkequal(String("42"), String("%d"), "__mod__", 42.0)
    checkequal(slongvalue, String("%d"), "__mod__", longvalue)
    checkcall(String("%d"), "__mod__", float(longvalue))
    checkequal(String("0042.00"), String("%07.2f"), "__mod__", 42)
    checkequal(String("0042.00"), String("%07.2F"), "__mod__", 42)

    checkraises(TypeError, String("abc"), "__mod__")
    checkraises(TypeError, String("%(foo)s"), "__mod__", 42)
    checkraises(TypeError, String("%s%s"), "__mod__", (42,))
    checkraises(TypeError, String("%c"), "__mod__", (None,))
    checkraises(ValueError, String("%(foo"), "__mod__", {})
    checkraises(
        TypeError,
        String("%(foo)s %(bar)s"),
        "__mod__",
        (String("foo"), 42),
    )
    checkraises(TypeError, String("%d"), "__mod__", String("42"))  # not numeric
    checkraises(
        TypeError, String("%d"), "__mod__", (42 + 0j)
    )  # no int conversion provided

    # argument names with properly nested brackets are supported
    checkequal(
        String("bar"),
        String("%((foo))s"),
        "__mod__",
        {String("(foo)"): String("bar")},
    )

    # 100 is a magic number in PyUnicode_Format, this forces a resize
    checkequal(
        103 * String("a") + String("x"),
        String("%sx"),
        "__mod__",
        103 * String("a"),
    )

    checkraises(
        TypeError,
        String("%*s"),
        "__mod__",
        (String("foo"), String("bar")),
    )
    checkraises(TypeError, String("%10.*f"), "__mod__", (String("foo"), 42.0))
    checkraises(ValueError, String("%10"), "__mod__", (42,))

    # Outrageously large width or precision should raise ValueError.
    checkraises(ValueError, String("%%%df") % (2 ** 64), "__mod__", (3.2))
    checkraises(ValueError, String("%%.%df") % (2 ** 64), "__mod__", (3.2))
    checkraises(
        OverflowError,
        String("%*s"),
        "__mod__",
        (sys.maxsize + 1, String("")),
    )
    checkraises(OverflowError, String("%.*f"), "__mod__", (sys.maxsize + 1, 1.0 / 7))

    class X(object):
        pass

    checkraises(TypeError, String("abc"), "__mod__", X())


@support.cpython_only
def test_formatting_c_limits():
    # third party
    from _testcapi import INT_MAX
    from _testcapi import PY_SSIZE_T_MAX
    from _testcapi import UINT_MAX

    SIZE_MAX = (1 << (PY_SSIZE_T_MAX.bit_length() + 1)) - 1
    checkraises(
        OverflowError,
        String("%*s"),
        "__mod__",
        (PY_SSIZE_T_MAX + 1, String("")),
    )
    checkraises(OverflowError, String("%.*f"), "__mod__", (INT_MAX + 1, 1.0 / 7))
    # Issue 15989
    checkraises(
        OverflowError,
        String("%*s"),
        "__mod__",
        (SIZE_MAX + 1, String("")),
    )
    checkraises(OverflowError, String("%.*f"), "__mod__", (UINT_MAX + 1, 1.0 / 7))


@pytest.mark.slow
def test_floatformatting():
    # float formatting
    for prec in range(100):
        format = String("%%.%if") % prec
        value = 0.01
        for x in range(60):
            value = value * 3.14159265359 / 3.0 * 10.0
            checkcall(format, "__mod__", value)


def test_inplace_rewrites():
    # Check that strings don't copy and modify cached single-character strings
    checkequal(String("a"), String("A"), "lower")
    checkequal(True, String("A"), "isupper")
    checkequal(String("A"), String("a"), "upper")
    checkequal(True, String("a"), "islower")

    checkequal(
        String("a"),
        String("A"),
        "replace",
        String("A"),
        String("a"),
    )
    checkequal(True, String("A"), "isupper")

    checkequal(String("A"), String("a"), "capitalize")
    checkequal(True, String("a"), "islower")

    checkequal(String("A"), String("a"), "swapcase")
    checkequal(True, String("a"), "islower")

    checkequal(String("A"), String("a"), "title")
    checkequal(True, String("a"), "islower")


def test_partition():
    checkequal(
        (
            String("this is the par"),
            String("ti"),
            String("tion method"),
        ),
        String("this is the partition method"),
        "partition",
        String("ti"),
    )

    # from raymond's original specification
    S = String("http://www.python.org")
    checkequal(
        (String("http"), String("://"), String("www.python.org")),
        S,
        "partition",
        String("://"),
    )
    checkequal(
        (String("http://www.python.org"), String(""), String("")),
        S,
        "partition",
        String("?"),
    )
    checkequal(
        (String(""), String("http://"), String("www.python.org")),
        S,
        "partition",
        String("http://"),
    )
    checkequal(
        (String("http://www.python."), String("org"), String("")),
        S,
        "partition",
        String("org"),
    )

    checkraises(ValueError, S, "partition", String(""))
    checkraises(TypeError, S, "partition", None)


def test_rpartition():
    checkequal(
        (
            String("this is the rparti"),
            String("ti"),
            String("on method"),
        ),
        String("this is the rpartition method"),
        "rpartition",
        String("ti"),
    )

    # from raymond's original specification
    S = String("http://www.python.org")
    checkequal(
        (String("http"), String("://"), String("www.python.org")),
        S,
        "rpartition",
        String("://"),
    )
    checkequal(
        (String(""), String(""), String("http://www.python.org")),
        S,
        "rpartition",
        String("?"),
    )
    checkequal(
        (String(""), String("http://"), String("www.python.org")),
        S,
        "rpartition",
        String("http://"),
    )
    checkequal(
        (String("http://www.python."), String("org"), String("")),
        S,
        "rpartition",
        String("org"),
    )

    checkraises(ValueError, S, "rpartition", String(""))
    checkraises(TypeError, S, "rpartition", None)


def test_none_arguments():
    # issue 11828
    s = String("hello")
    checkequal(2, s, "find", String("l"), None)
    checkequal(3, s, "find", String("l"), -2, None)
    checkequal(2, s, "find", String("l"), None, -2)
    checkequal(0, s, "find", String("h"), None, None)

    checkequal(3, s, "rfind", String("l"), None)
    checkequal(3, s, "rfind", String("l"), -2, None)
    checkequal(2, s, "rfind", String("l"), None, -2)
    checkequal(0, s, "rfind", String("h"), None, None)

    checkequal(2, s, "index", String("l"), None)
    checkequal(3, s, "index", String("l"), -2, None)
    checkequal(2, s, "index", String("l"), None, -2)
    checkequal(0, s, "index", String("h"), None, None)

    checkequal(3, s, "rindex", String("l"), None)
    checkequal(3, s, "rindex", String("l"), -2, None)
    checkequal(2, s, "rindex", String("l"), None, -2)
    checkequal(0, s, "rindex", String("h"), None, None)

    checkequal(2, s, "count", String("l"), None)
    checkequal(1, s, "count", String("l"), -2, None)
    checkequal(1, s, "count", String("l"), None, -2)
    checkequal(0, s, "count", String("x"), None, None)

    checkequal(True, s, "endswith", String("o"), None)
    checkequal(True, s, "endswith", String("lo"), -2, None)
    checkequal(True, s, "endswith", String("l"), None, -2)
    checkequal(False, s, "endswith", String("x"), None, None)

    checkequal(True, s, "startswith", String("h"), None)
    checkequal(True, s, "startswith", String("l"), -2, None)
    checkequal(True, s, "startswith", String("h"), None, -2)
    checkequal(False, s, "startswith", String("x"), None, None)
