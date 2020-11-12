# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
import pytest

# syft absolute
from syft.decorators import syft_decorator


def test_typecheck_basic_dtypes() -> None:
    @syft_decorator(typechecking=True)
    def func(x: int, y: int) -> int:
        return x + y

    func(x=1, y=2)

    with pytest.raises(TypeError) as e:
        func(x="test", y=2)

    assert str(e.value) == 'type of argument "x" must be int; got str instead'


def test_typecheck_generic_dtypes() -> None:
    @syft_decorator(typechecking=True)
    def func(x: List[str], y: Union[List[str], List[int]]) -> int:
        return 0

    func(x=["1", "2", "3"], y=[1, 2, 3])
    func(x=["2", "2", "3"], y=["unu", "doi", "trei"])

    with pytest.raises(TypeError) as e:
        func(x=[1, 2, 3], y=["unu", "doi", "trei"])

    assert str(e.value) == 'type of argument "x"[0] must be str; got int instead'

    with pytest.raises(TypeError) as e:
        func(x=["1", "2", "3"], y=[1, 2, 2.0])


def test_optional() -> None:
    @syft_decorator(typechecking=True)
    def func(x: Optional[int]) -> int:
        return 0

    func(x=0)
    func(x=None)

    with pytest.raises(TypeError):
        func(x="test")


def test_mappings() -> None:
    @syft_decorator(typechecking=True)
    def func(x: Dict[str, str]) -> int:
        return 0

    func(x={"1": "2"})

    with pytest.raises(TypeError) as e:
        func(x={1: "1"})

    assert str(e.value) == 'type of keys of argument "x" must be str; got int instead'


def test_ret_type() -> None:
    @syft_decorator(typechecking=True)
    def func() -> int:
        return 0

    func()

    with pytest.raises(TypeError) as e:

        @syft_decorator(typechecking=True)
        def func() -> int:
            return 1.0  # type: ignore

        func()

    assert str(e.value) == "type of the return value must be int; got float instead"
