from syft.ast.globals import Globals

# syft relative
from . import module_test


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


# -------------------- Module Tests --------------------


def test_module_repr() -> None:
    ast = Globals(None)

    for method, return_type in methods:
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

    for method, return_type in methods:
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
