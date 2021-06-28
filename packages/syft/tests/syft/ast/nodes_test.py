# stdlib
from functools import partial

# third party
import pytest
from pytest import CaptureFixture

# syft absolute
import syft
from syft.ast.globals import Globals
from syft.core.node.common.client import Client
from syft.lib import lib_ast

# syft relative
from . import module_test
from .functionality_test import module_test_methods
from .functionality_test import update_ast_test

iter_without_len_methods = [
    ("module_test.IterWithoutLen", "module_test.IterWithoutLen"),
    ("module_test.IterWithoutLen.__iter__", "syft.lib.python.Iterator"),
    ("module_test.IterWithoutLen.__next__", "syft.lib.python.Int"),
]


@pytest.fixture(scope="function")
def register_module_test_iter_without_len() -> None:
    # Make lib_ast contain the specific methods/attributes
    update_ast_test(ast_or_client=syft.lib_ast, methods=iter_without_len_methods)

    # Make sure that when we register a new client it would update the specific AST
    lib_ast.loaded_lib_constructors["module_test"] = partial(
        update_ast_test, methods=iter_without_len_methods
    )


@pytest.fixture()
def custom_client() -> Client:
    alice = syft.VirtualMachine(name="alice")
    alice_client = alice.get_root_client()

    return alice_client


# -------------------- Module Tests --------------------


def test_module_repr() -> None:
    ast = Globals(None)

    for method, return_type in module_test_methods:
        ast.add_path(
            path=method, framework_reference=module_test, return_type_name=return_type
        )

    expected_repr = "Module:\n"
    for name, module in ast.attrs.items():
        expected_repr += (
            "\t." + name + " -> " + str(module).replace("\t.", "\t\t.") + "\n"
        )

    assert ast.__repr__() == expected_repr


# -------------------- Klass Tests --------------------


def test_klass_get_and_set_request_config() -> None:
    ast = Globals(None)

    for method, return_type in module_test_methods:
        ast.add_path(
            path=method, framework_reference=module_test, return_type_name=return_type
        )

    for klass in ast.classes:
        klass.create_pointer_class()
        klass_ptr = klass.pointer_type

        # Test Klass pointer has get_request_config attribute
        get_request_config = klass.pointer_type.get_request_config
        assert get_request_config(klass_ptr) == {
            "request_block": True,
            "timeout_secs": 25,
            "delete_obj": False,
        }

        # Test Klass pointer has set_request_config attribute and it works
        set_request_config = klass.pointer_type.set_request_config
        set_request_config(
            klass_ptr,
            {
                "request_block": False,
                "timeout_secs": 25,
                "delete_obj": True,
            },
        )
        get_request_config = klass.pointer_type.get_request_config
        assert get_request_config() == {
            "request_block": False,
            "timeout_secs": 25,
            "delete_obj": True,
        }


def test_klass_wrap_iterator_raises_exception(
    register_module_test_iter_without_len: CaptureFixture, custom_client: Client
) -> None:
    iter_without_len_ptr = custom_client.module_test.IterWithoutLen()
    with pytest.raises(ValueError) as exception_info:
        iter_without_len_ptr.__iter__()
    assert (
        str(exception_info.value)
        == "Can't build a remote iterator on an object with no __len__."
    )
