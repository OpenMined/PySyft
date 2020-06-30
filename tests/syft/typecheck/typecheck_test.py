import pytest
from typing import List, Union, Optional, Dict
from syft.typecheck.typecheck import type_hints


def test_typecheck_basic_dtypes():
    @type_hints
    def func(x: int, y: int) -> int:
        return x + y

    func(x=1, y=2)

    with pytest.raises(AttributeError) as e:
        func(x="test", y=2)

    assert (
        str(e.value) == "Error in argument x: Argument should have type <class 'int'>."
    )


def test_typecheck_generic_dtypes():
    @type_hints
    def func(x: List[str], y: Union[List[str], List[int]]) -> int:
        return 0

    func(x=["1", "2", "3"], y=[1, 2, 3])
    func(x=["2", "2", "3"], y=["unu", "doi", "trei"])

    with pytest.raises(AttributeError) as e:
        func(x=[1, 2, 3], y=["unu", "doi", "trei"])

    assert (
        str(e.value)
        == "Error in argument x: Iterable should have type typing.List[str]."
    )

    with pytest.raises(AttributeError) as e:
        func(x=["1", "2", "3"], y=[1, 2, 2.0])

    assert (
        str(e.value) == "Error in argument y: Argument should have any of the types "
        "(typing.List[str], typing.List[int])."
    )


def test_optional():
    @type_hints
    def func(x: Optional[int]) -> int:
        return 0

    func(x=0)
    func(x=None)

    with pytest.raises(AttributeError) as _:
        func(x="test")


def test_mappings():
    @type_hints
    def func(x: Dict[str, str]) -> int:
        return 0

    func(x={"1": "2"})

    with pytest.raises(AttributeError) as e:
        func(x={1: "1"})

    assert (
        str(e.value)
        == "Error in argument x: Key element of mapping should have type <class 'str'>."
    )


def test_ret_type():
    @type_hints
    def func() -> int:
        return 0

    func()

    with pytest.raises(AttributeError) as e:

        @type_hints
        def func() -> int:
            return 1.0

        func()

    assert str(e.value) == "Return type is <class 'float'>, should be <class 'int'>."
