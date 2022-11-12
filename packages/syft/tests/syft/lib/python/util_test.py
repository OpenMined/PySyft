# syft absolute
from syft.common.lib_ast_shares import downcast_args_and_kwargs
from syft.lib.python.primitive_factory import PrimitiveFactory


def test_downcast():
    assert downcast_args_and_kwargs(args=[1, 2, 3], kwargs={}) == ([1, 2, 3], {})
    assert downcast_args_and_kwargs(args=["test", True], kwargs={1: 2}) == (
        ["test", True],
        {1: 2},
    )


def test_recurse():
    assert PrimitiveFactory.generate_primitive(value={1: 1, 2: 2}, recurse=True) == {
        1: 1,
        2: 2,
    }
