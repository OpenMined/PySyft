# third party
import pytest

# syft absolute
from syft.service.action.action_object import ActionObject
from syft.types.result import Err
from syft.types.result import Ok
from syft.types.result import as_result


def test_ok():
    good = Ok("om")

    assert good.is_ok() is True
    assert good.is_err() is False
    assert good.ok() == "om"


def test_ok_is_not_err():
    good = Ok("om")

    assert good.is_err() is False
    assert good.err() is None
    assert good.unwrap() == "om"


def test_ok_value_property():
    good = Ok("om")

    assert good.ok_value == "om"


def test_ok_match():
    matched = Ok(True)

    match matched:
        case Ok(x):
            assert x is True


def test_error():
    bad = Err(OSError("some exception"))

    assert bad.is_ok() is False
    assert bad.is_err() is True
    assert type(bad.err()) is OSError


def test_err_is_not_ok():
    bad = Err(OSError("some exception"))

    assert bad.is_ok() is False
    assert bad.ok() is None


def test_err_value_property():
    bad = Err(OSError("some exception"))

    assert type(bad.error_value) is OSError
    assert bad.error_value.args == ("some exception",)


def test_err_match():
    matched = Err(OSError("some exception"))

    match matched:
        case Err(e):
            assert type(e) is OSError
            assert e.args == ("some exception",)


def test_unwrap_ok():
    obj = ActionObject.from_obj("om")
    result = Ok(obj)

    same_obj = result.unwrap()
    assert same_obj == obj


def test_unwrap_error():
    result = Err(ValueError("some exception"))

    with pytest.raises(ValueError):
        result.unwrap()


def test_unwrap_error_not_exception():
    str_ = "some_exception"
    result = Err(str_)  # type: ignore

    with pytest.raises(TypeError):
        result.unwrap()


def test_as_result_decorator_good():
    @as_result(ValueError)
    def good() -> str:
        return "om"

    result = good()

    assert result.is_ok() is True
    assert result.is_err() is False
    assert result.ok() == "om"
    assert result.unwrap() == "om"


def test_as_result_decorator_bad():
    @as_result(ValueError)
    def bad() -> str:
        raise ValueError("some exception")

    result = bad()

    assert result.is_err() is True
    assert result.is_ok() is False

    e = result.err()
    assert type(e) is ValueError
    assert e.args == ("some exception",)

    with pytest.raises(ValueError):
        result.unwrap()


def test_as_result_decorator():
    @as_result(ValueError)
    def create_object(valid: bool) -> ActionObject:
        if valid:
            return ActionObject.from_obj("om")
        else:
            raise ValueError("some exception")

    result = create_object(True)

    assert result.is_ok() is True
    assert result.is_err() is False

    obj = result.unwrap()
    assert isinstance(obj, ActionObject)
    assert obj.syft_action_data == "om"

    result = create_object(False)

    assert result.is_err() is True
    assert result.is_ok() is False
    assert type(result.err()) is ValueError

    with pytest.raises(ValueError):
        result.unwrap()


def test_as_result_decorator_bubble_up():
    @as_result(ValueError, TypeError)
    def more_decorators(a: int) -> str:
        if a == 1:
            return "om"
        raise OSError("some exception")

    result = more_decorators(1)
    assert result.is_ok() is True
    assert result.ok() == "om"

    with pytest.raises(OSError):
        more_decorators(0)


def test_as_result_decorator_multiple_exceptions():
    @as_result(ValueError, TypeError, OSError)
    def multiple_exceptions(a: int) -> str:
        if a == 1:
            return "om"
        if a == 2:
            raise TypeError
        if a == 3:
            raise ValueError
        if a == 4:
            raise OSError
        raise ArithmeticError

    result = multiple_exceptions(1)
    assert result.ok() == "om"

    result_type = multiple_exceptions(2)
    assert type(result_type.err()) is TypeError

    result_value = multiple_exceptions(3)
    assert type(result_value.err()) is ValueError

    result_os = multiple_exceptions(4)
    assert type(result_os.err()) is OSError

    with pytest.raises(ArithmeticError):
        multiple_exceptions(5)


def test_as_result_decorator_sub():
    class TestException(Exception):
        pass

    class SubException(TestException):
        pass

    @as_result(TestException)
    def subclassed() -> str:
        raise SubException("some exception")

    result = subclassed()

    assert result.is_err() is True
    assert result.is_ok() is False

    assert type(result.err()) is SubException

    with pytest.raises(SubException):
        result.unwrap()
