# stdlib
from types import ModuleType

# syft absolute
from syft.types.errors import ExceptionFilter


def create_empty_module(module_name: str):
    code = """
class NonExceptionClass: ...
    """.strip()
    return create_module(module_name, code=code)


def create_module(module_name: str, code: str | None = None):
    # stdlib
    import sys

    created_module = ModuleType(module_name)

    module_code = (
        code
        or """
class CustomException(Exception): ...
class AnotherCustomException(Exception): ...
class InheritedException(CustomException): ...
class NonExceptionClass: ...
    """
    )

    exec(module_code, created_module.__dict__)

    sys.modules[module_name] = created_module

    return created_module


def test_exception_filter_init():
    instance = ExceptionFilter("pydantic")

    assert isinstance(instance, ExceptionFilter)
    assert isinstance(instance, tuple)
    assert instance.module == "pydantic"
    assert instance


def test_exception_filter_exceptions():
    module_name = "test_module"
    module = create_module(module_name)

    instance = ExceptionFilter(module_name)

    # classes are sorted by name
    assert instance == (
        module.AnotherCustomException,
        module.CustomException,
        module.InheritedException,
    )


def test_not_found_module_doesnt_crash():
    instance = ExceptionFilter("fake_syft_module")

    assert instance == ()


def test_exception_filter_no_exceptions():
    module_name = "syft_test_empty_module"

    create_empty_module(module_name)
    instance = ExceptionFilter(module=module_name)

    assert tuple(instance) == ()
