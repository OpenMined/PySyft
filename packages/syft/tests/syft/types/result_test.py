# third party
import pytest

# syft absolute
from syft import ActionObject
from syft.types.result import Error
from syft.types.result import Ok


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
    bad = Error(OSError("some exception"))

    assert bad.is_ok() is False
    assert bad.is_err() is True
    assert type(bad.err()) is OSError


def test_err_is_not_ok():
    bad = Error(OSError("some exception"))

    assert bad.is_ok() is False
    assert bad.ok() is None


def test_err_value_property():
    bad = Error(OSError("some exception"))

    assert type(bad.error_value) is OSError
    assert bad.error_value.args == ("some exception",)


def test_err_match():
    matched = Error(OSError("some exception"))

    match matched:
        case Error(e):
            assert type(e) is OSError
            assert e.args == ("some exception",)


def test_unwrap_ok():
    obj = ActionObject.from_obj("om")
    result = Ok(obj)

    same_obj = result.unwrap()
    assert same_obj == obj


def test_unwrap_error():
    result = Error(ValueError("some exception"))

    with pytest.raises(ValueError):
        result.unwrap()


def test_unwrap_error_not_exception():
    str_ = "some_exception"
    result = Error(str_)  # type: ignore

    with pytest.raises(TypeError):
        result.unwrap()
