# syft absolute
import syft as sy
from syft.ast.util import unsplit


def test_util_unsplit() -> None:
    fqn_list = unsplit(["syft", "lib", "python", "List"])

    assert fqn_list == "syft.lib.python.List"


def test_path_cache() -> None:
    short_fqn_list = "syft.lib.python.List"
    long_fqn_list = "syft.lib.python.list.List"

    list_ref1 = sy.lib_ast(
        short_fqn_list, return_callable=True, obj_type=sy.lib.python.List
    )

    list_ref2 = sy.lib_ast(
        long_fqn_list, return_callable=True, obj_type=sy.lib.python.List
    )

    assert list_ref1.ref == sy.lib.python.List
    assert list_ref1.name == "List"
    assert list_ref1.path_and_name == short_fqn_list

    assert list_ref2.ref == sy.lib.python.List
    assert list_ref2.name == "List"
    assert list_ref2.path_and_name == short_fqn_list

    assert list_ref1 == list_ref2
