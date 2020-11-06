# syft absolute
from syft.ast.util import unsplit


def test_util_unsplit() -> None:
    fqn_list = unsplit(["syft", "lib", "python", "List"])

    assert fqn_list == "syft.lib.python.List"
