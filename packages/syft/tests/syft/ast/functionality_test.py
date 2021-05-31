"""
The following test suit serves as a set of examples of how to integrate different classes
into our AST and use them.
"""
# stdlib
from functools import partial
from importlib import reload
import sys
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union as TypeUnion

# third party
import pytest

# syft absolute
import syft
from syft.ast import add_dynamic_objects
from syft.ast.globals import Globals
from syft.core.node.abstract.node import AbstractNodeClient
from syft.core.node.common.client import Client
from syft.lib import lib_ast

# syft relative
from . import module_test

sys.modules["module_test"] = module_test

module_test_methods = [
    ("module_test.A", "module_test.A"),
    ("module_test.A.__len__", "syft.lib.python.Int"),
    ("module_test.A.__iter__", "syft.lib.python.Iterator"),
    ("module_test.A.__next__", "syft.lib.python.Int"),
    ("module_test.A.test_method", "syft.lib.python.Int"),
    ("module_test.A.test_property", "syft.lib.python.Float"),
    ("module_test.A._private_attr", "syft.lib.python.Float"),
    ("module_test.A.static_method", "syft.lib.python.Float"),
    ("module_test.A.static_attr", "syft.lib.python.Int"),
    ("module_test.B.Car", "module_test.B"),
    ("module_test.C", "module_test.C"),
    ("module_test.C.type_reload_func", "syft.lib.python._SyNone"),
    ("module_test.C.obj_reload_func", "syft.lib.python._SyNone"),
    ("module_test.C.dummy_reloadable_func", "syft.lib.python.Int"),
    ("module_test.global_value", "syft.lib.python.Int"),
    ("module_test.global_function", "syft.lib.python.Int"),
]

dynamic_objects = [("module_test.C.dynamic_object", "syft.lib.python.Int")]


def update_ast_test(
    ast_or_client: TypeUnion[Globals, AbstractNodeClient],
    methods: List[Tuple[str, str]],
    dynamic_objects: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Checks functionality of update_ast, uses create_ast"""
    if isinstance(ast_or_client, Globals):
        ast = ast_or_client
        test_ast = create_ast_test(
            client=None, methods=methods, dynamic_objects=dynamic_objects
        )
        ast.add_attr(attr_name="module_test", attr=test_ast.attrs["module_test"])
    elif isinstance(ast_or_client, AbstractNodeClient):
        client = ast_or_client
        test_ast = create_ast_test(
            client=client, methods=methods, dynamic_objects=dynamic_objects
        )
        client.lib_ast.attrs["module_test"] = test_ast.attrs["module_test"]
        setattr(client, "module_test", test_ast.attrs["module_test"])
    else:
        raise ValueError(
            f"Expected param of type (Globals, AbstractNodeClient), but got {type(ast_or_client)}"
        )


def create_ast_test(
    client: Optional[AbstractNodeClient],
    methods: List[Tuple[str, str]],
    dynamic_objects: Optional[List[Tuple[str, str]]],
) -> Globals:
    """Unit test for create_ast functionality"""
    ast = Globals(client)

    for method, return_type in methods:
        ast.add_path(
            path=method, framework_reference=module_test, return_type_name=return_type
        )

    if dynamic_objects:
        add_dynamic_objects(ast, dynamic_objects)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_storable_object_attr_convenience_methods()

    return ast


@pytest.fixture(autouse=True, scope="module")
def register_module_test() -> None:
    """Test which is required for every other tests (runs first even in random execution)"""
    # Make lib_ast contain the specific methods/attributes
    update_ast_test(
        ast_or_client=syft.lib_ast,
        methods=module_test_methods,
        dynamic_objects=dynamic_objects,
    )

    # Make sure that when we register a new client it would update the specific AST
    lib_ast.loaded_lib_constructors["module_test"] = partial(
        update_ast_test, methods=module_test_methods, dynamic_objects=dynamic_objects
    )


@pytest.fixture()
def custom_client() -> Client:
    """Return VM for unit tests"""
    alice = syft.VirtualMachine(name="alice")
    alice_client = alice.get_root_client()

    return alice_client


def test_len(custom_client: Client) -> None:
    """Unit test to check length of the class"""
    a_ptr = custom_client.module_test.A()
    result_from_ptr = a_ptr.__len__()

    a = module_test.A()
    result = len(a)

    assert result == result_from_ptr


def test_iter(custom_client: Client) -> None:
    """Unit test to check iterator of the class"""
    a_ptr = custom_client.module_test.A()
    iter_from_ptr = a_ptr.__iter__()
    a = module_test.A()
    iter_from_obj = iter(a)
    for _ in range(1, len(a)):
        assert next(iter_from_ptr).get() == next(iter_from_obj)


def test_method(custom_client: Client) -> None:
    """Unit test to check method of remote class object"""
    a_ptr = custom_client.module_test.A()
    result_ptr = a_ptr.test_method()

    a = module_test.A()
    result = a.test_method()

    assert result == result_ptr.get()


def test_property_get(custom_client: Client) -> None:
    """Unit test to check property(get) of remote class object"""
    a_ptr = custom_client.module_test.A()
    result_ptr = a_ptr.test_property

    a = module_test.A()
    result = a.test_property

    assert result == result_ptr.get()


def test_property_set(custom_client: Client) -> None:
    """Unit test to check property(set) of remote class object"""
    value_to_set = 7.5

    a_ptr = custom_client.module_test.A()
    a_ptr.test_property = value_to_set
    result_ptr = a_ptr.test_property

    a = module_test.A()
    a.test_property = value_to_set
    result = a.test_property

    assert result == result_ptr.get()  # type: ignore


@pytest.mark.xfail(strict=False)
def test_slot_get(custom_client: Client) -> None:
    """Unit test to check slot(get) of remote class object"""
    a_ptr = custom_client.module_test.A()
    result_ptr = a_ptr._private_attr

    a = module_test.A()
    result = a._private_attr

    assert result == result_ptr.get()


@pytest.mark.xfail(strict=False)
def test_slot_set(custom_client: Client) -> None:
    """Unit test to check property(set) of remote class object"""
    value_to_set = 7.5

    a_ptr = custom_client.module_test.A()
    a_ptr._private_attr = value_to_set
    result_ptr = a_ptr._private_attr

    a = module_test.A()
    a._private_attr = value_to_set
    result = a._private_attr

    assert result == result_ptr.get()  # type: ignore


def test_global_function(custom_client: Client) -> None:
    """Unit test to check global function of remote class object"""
    result_ptr = custom_client.module_test.global_function()
    result = module_test.global_function()

    assert result == result_ptr.get()


def test_global_attribute_get(custom_client: Client) -> None:
    """Unit test to check global attribute(get) of remote class object"""
    result_ptr = custom_client.module_test.global_value
    result = module_test.global_value

    assert result == result_ptr.get()


def test_global_attribute_set(custom_client: Client) -> None:
    """Unit test to check global attribute(set) of remote class object"""
    global module_test

    set_value = 5

    custom_client.module_test.global_value = set_value
    result_ptr = custom_client.module_test.global_value
    sy_result = result_ptr.get()  # type: ignore

    module_test = reload(module_test)
    module_test.global_value = set_value
    local_result = module_test.global_value

    assert local_result == sy_result


def test_static_method(custom_client: Client) -> None:
    """Unit test to check static of remote class object"""
    result_ptr = custom_client.module_test.A.static_method()
    result = module_test.A.static_method()
    assert result == result_ptr.get()


def test_static_attribute_set_get(custom_client: Client) -> None:

    """Unit test to check static_attribute(get & set) of remote class object"""
    result_ptr = custom_client.module_test.A.static_attr
    result = module_test.A.static_attr

    assert result == result_ptr.get()

    value_to_set = 5

    custom_client.module_test.A.static_attr = value_to_set
    result_ptr = custom_client.module_test.A.static_attr

    module_test.A.static_attr = value_to_set
    result = module_test.A.static_attr

    assert result == result_ptr.get()


def test_enum(custom_client: Client) -> None:
    """Unit test for enum class"""
    result_ptr = custom_client.module_test.B.Car
    result = module_test.B.Car

    assert result == result_ptr.get()


def test_dynamic_ast_type(custom_client: Client) -> None:
    """Unit test for dynamic ast for remote class"""
    custom_client.module_test.C.type_reload_func()
    obj_ptr = custom_client.module_test.C()
    result_ptr = obj_ptr.dummy_reloadable_func()

    assert result_ptr.get() == 1


def test_dynamic_ast_obj(custom_client: Client) -> None:
    """Unit test for dynamic ast(object) for remote class"""
    obj_ptr = custom_client.module_test.C()
    obj_ptr.obj_reload_func()
    result_ptr = obj_ptr.dummy_reloadable_func()

    assert result_ptr.get() == 2


def test_dynamic_object_get(custom_client: Client) -> None:
    obj_ptr = custom_client.module_test.C().dynamic_object
    obj = module_test.C().dynamic_object

    assert obj_ptr.get() == obj


def test_dynamic_object_set(custom_client: Client) -> None:
    value = 0
    obj_ptr = custom_client.module_test.C()
    obj_ptr.dynamic_object = value

    assert obj_ptr.dynamic_object.get() == 0  # type: ignore
