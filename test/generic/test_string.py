from syft.generic.string import String
import syft as sy


def test_string_methods():
    """
        Tests some of the `String` methods which are hooked from `str`.
    more tests are to be added
    """

    # Create a string
    string = String("Hello PySyft")

    assert isinstance(string.upper(), String)
    assert isinstance(string.lower(), String)
    assert isinstance(string.title(), String)

    assert string == "Hello PySyft"
    assert string == String("Hello PySyft")

    assert string.upper() == "HELLO PYSYFT"
    assert string.upper() == String("HELLO PYSYFT")

    assert string.lower() == "hello pysyft"
    assert string.lower() == String("hello pysyft")

    assert string.title() == "Hello Pysyft"
    assert string.title() == String("Hello Pysyft")
    assert string.title() >= String("Hello Pysyft")
    assert string.title() <= String("Hello Pysyft")

    assert string.startswith("Hel") is True
    assert string.startswith(String("Hel")) is True

    assert string.endswith("Syft") is True
    assert string.endswith(String("Syft")) is True

    assert (string > "Hello PySyfa") is True
    assert (string >= "Hello PySyfa") is True

    assert (string < "Hello PySyfz") is True
    assert (string <= "Hello PySyfz") is True

    assert String(" Hello").lstrip() == "Hello"
    assert String("Hello ").rstrip() == "Hello"

    assert String("Hello").center(9) == "  Hello  "
    assert String("Hello").center(9) == String("  Hello  ")

    assert String("Hello").rjust(10) == "     Hello"
    assert String("Hello").rjust(10) == String("     Hello")

    assert String("Hello").ljust(10) == "Hello     "
    assert String("Hello").ljust(10) == String("Hello     ")

    assert string + string == "Hello PySyftHello PySyft"
    assert isinstance(string + string, String)
    assert isinstance(string + " !", String)

    assert f"{string} !" == "Hello PySyft !"

    assert String("Hello {}").format(String("PySyft")) == string
    assert String("Hello %s") % "PySyft" == string

    assert str(string) == "Hello PySyft"

    x = String("Hello PySyft")
    bob = sy.VirtualWorker(id="bob", hook=sy.hook)
    out = x.send(bob)
    assert isinstance(out, sy.generic.pointers.string_pointer.StringPointer)
