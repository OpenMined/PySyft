# syft absolute
import syft as sy


def test_union_pointer_creation() -> None:
    # check that we don't have this random type in our ast
    assert not hasattr(sy.lib_ast.syft.lib.misc.union, "BoolListUnion")

    union_pointer_type = sy.lib.misc.union.UnionGenerator[
        "syft.lib.python.Bool", "syft.lib.python.List"
    ]

    # for a union type to be generated in the ast, we need to rebuild it.
    new_ast = sy.lib.create_lib_ast()
    assert hasattr(new_ast.syft.lib.misc.union, "BoolListUnion")

    # the newly added type is visible in the global scope
    union_type = getattr(
        sy.lib.misc.union, union_pointer_type.rsplit(".", maxsplit=1)[-1]
    )

    # the union has all common allowed functions
    assert hasattr(union_type, "__add__")

    # the union respects List specific operations that don't interfere
    # with the Bool method set
    assert hasattr(union_type, "append")
    assert hasattr(union_type, "clear")

    # the union respects Bool specific operations that don't interfere
    # with the List method set
    assert hasattr(union_type, "__float__")
    assert hasattr(union_type, "__pow__")

    # inspecting that function generation is respected in the pointer type as
    # well, we care only about the pointer type. The union is acting as a
    # proxy to the original type, it's not going to be created.
    union_pointer_function_set = new_ast.syft.lib.misc.union.BoolListUnion.attrs.keys()
    bool_function_set = set(new_ast.syft.lib.python.Bool.attrs.keys())
    list_function_set = set(new_ast.syft.lib.python.List.attrs.keys())
    bool_list_function_set = bool_function_set.union(list_function_set)

    assert "__add__" in union_pointer_function_set
    assert "append" in union_pointer_function_set
    assert "clear" in union_pointer_function_set
    assert "__float__" in union_pointer_function_set
    assert "__pow__" in union_pointer_function_set

    # dummy example found out in the codebase of the behavior of a denied function
    # in the union type -> `__radd__`

    # function present in the Bool pointer
    assert "__radd__" in bool_list_function_set

    # function not present in the List pointer
    assert "__radd__" not in list_function_set

    # function present in the List function set
    assert "__radd__" in dir(sy.lib.python.List)

    # the __radd__ is denied from the List pointer function set
    # thus, neither the union pointer should have it due to possible anomalous
    # behaviors. If you want to use it on Bool, you will need to cast it.
    assert "__radd__" not in union_pointer_function_set
