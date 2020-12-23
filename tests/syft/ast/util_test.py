# syft absolute
import syft as sy


def test_path_cache() -> None:
    short_fqn_list = "syft.lib.python.List"
    long_fqn_list = "syft.lib.python.list.List"

    list_ref1 = sy.lib_ast.query(
        short_fqn_list
    )

    list_ref2 = sy.lib_ast.query(
        long_fqn_list
    )

    assert list_ref1.ref == sy.lib.python.List
    assert list_ref1.name == "List"
    assert list_ref1.path_and_name == short_fqn_list

    assert list_ref2.ref == sy.lib.python.List
    assert list_ref2.name == "List"
    assert list_ref2.path_and_name == short_fqn_list

    assert list_ref1 == list_ref2
