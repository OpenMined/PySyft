"""
The following test suit serves as a set of examples of how to integrate different classes
into our AST and use them.
"""
# stdlib
from importlib import reload
from typing import Any as TypeAny
from typing import Union as TypeUnion

# third party
import pytest

# syft absolute
import syft
from syft.ast.globals import Globals
from syft.core.node.common.client import Client
from syft.lib import lib_ast

# syft relative
from . import module_test


def update_ast_test(ast: TypeUnion[Globals, TypeAny], client: TypeAny = None) -> None:
    test_ast = create_ast_test(client=client)
    ast.add_attr(attr_name="module_test", attr=test_ast.attrs["module_test"])


def create_ast_test(client: Client) -> Globals:
    ast = Globals(client)

    methods = [
        ("module_test.A", "module_test.A"),
        ("module_test.A.test_method", "syft.lib.python.Int"),
        ("module_test.A.test_property", "syft.lib.python.Float"),
        ("module_test.A._private_attr", "syft.lib.python.Float"),
        ("module_test.A.static_method", "syft.lib.python.Float"),
        ("module_test.A.static_attr", "syft.lib.python.Int"),
        ("module_test.B.Car", "module_test.B"),
        ("module_test.global_value", "syft.lib.python.Int"),
        ("module_test.global_function", "syft.lib.python.Int"),
    ]

    for method, return_type in methods:
        ast.add_path(
            path=method, framework_reference=module_test, return_type_name=return_type
        )

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast


@pytest.fixture(autouse=True, scope="module")
def registr_module_test() -> None:
    # Make lib_ast contain the specific methods/attributes
    update_ast_test(ast=syft.lib_ast)

    # Make sure that when we register a new client it would update the specific AST
    lib_ast.loaded_lib_constructors["module_test"] = update_ast_test


def get_custom_client() -> Client:
    alice = syft.VirtualMachine(name="alice")
    alice_client = alice.get_root_client()

    return alice_client


def test_method() -> None:
    client = get_custom_client()
    a_ptr = client.module_test.A()
    result_ptr = a_ptr.test_method()

    a = module_test.A()
    result = a.test_method()

    assert result == result_ptr.get()


def test_property_get() -> None:
    client = get_custom_client()
    a_ptr = client.module_test.A()
    result_ptr = a_ptr.test_property

    a = module_test.A()
    result = a.test_property

    assert result == result_ptr.get()


def test_property_set() -> None:
    value_to_set = 7.5
    client = get_custom_client()

    a_ptr = client.module_test.A()
    a_ptr.test_property = value_to_set
    result_ptr = a_ptr.test_property

    a = module_test.A()
    a.test_property = value_to_set
    result = a.test_property

    assert result == result_ptr.get()  # type: ignore


def test_slot_get() -> None:
    client = get_custom_client()

    a_ptr = client.module_test.A()
    result_ptr = a_ptr._private_attr

    a = module_test.A()
    result = a._private_attr

    assert result == result_ptr.get()


def test_slot_set() -> None:
    value_to_set = 7.5
    client = get_custom_client()

    a_ptr = client.module_test.A()
    a_ptr._private_attr = value_to_set
    result_ptr = a_ptr._private_attr

    a = module_test.A()
    a._private_attr = value_to_set
    result = a._private_attr

    assert result == result_ptr.get()  # type: ignore


def test_global_function() -> None:
    client = get_custom_client()

    result_ptr = client.module_test.global_function()
    result = module_test.global_function()

    assert result == result_ptr.get()


def test_global_attribute_get() -> None:
    client = get_custom_client()

    result_ptr = client.module_test.global_value
    result = module_test.global_value

    assert result == result_ptr.get()


def test_global_attribute_set() -> None:
    global module_test

    set_value = 5
    client = get_custom_client()

    client.module_test.global_value = set_value
    result_ptr = client.module_test.global_value
    sy_result = result_ptr.get()  # type: ignore

    module_test = reload(module_test)
    module_test.global_value = set_value
    local_result = module_test.global_value

    assert local_result == sy_result


def test_static_method() -> None:
    client = get_custom_client()

    result_ptr = client.module_test.A.static_method()
    result = module_test.A.static_method()
    assert result == result_ptr.get()


def test_static_attribute_get() -> None:
    client = get_custom_client()

    result_ptr = client.module_test.A.static_attr
    result = module_test.A.static_attr

    assert result == result_ptr.get()


def test_static_attribute_set() -> None:
    value_to_set = 5
    client = get_custom_client()

    client.module_test.A.static_attr = value_to_set
    result_ptr = client.module_test.A.static_attr

    module_test.A.static_attr = value_to_set
    result = module_test.A.static_attr

    assert result == result_ptr.get()  # type: ignore


def test_enum() -> None:
    client = get_custom_client()

    result_ptr = client.module_test.B.Car
    result = module_test.B.Car

    assert result == result_ptr.get()
