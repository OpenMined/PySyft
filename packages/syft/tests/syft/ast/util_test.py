# syft absolute
import syft as sy


def test_path_cache() -> None:
    short_fqn_list = "syft.lib.python.List"
    long_fqn_list = "syft.lib.python.list.List"

    list_ref1 = sy.lib_ast.query(short_fqn_list, obj_type=sy.lib.python.List)
    list_ref2 = sy.lib_ast.query(long_fqn_list, obj_type=sy.lib.python.List)
    list_ref2 = sy.lib_ast.query(long_fqn_list)

    assert list_ref1.object_ref == sy.lib.python.List
    assert list_ref1.name == "List"
    assert list_ref1.path_and_name == short_fqn_list

    assert list_ref2.object_ref == sy.lib.python.List
    assert list_ref2.name == "List"
    assert list_ref2.path_and_name == short_fqn_list

    assert list_ref1 == list_ref2
