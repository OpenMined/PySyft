# third party
import pytest

# syft absolute
from syft.types.tupledict import DictTuple

# different ways to create a DictTuple
TEST_CASES = [
    DictTuple({"x": 1, "y": 2}),
    DictTuple([("x", 1), ("y", 2)]),
    DictTuple([1, 2], ["x", "y"]),
]


@pytest.mark.parametrize("dict_tuple", TEST_CASES)
def test_should_iter_over_value(dict_tuple) -> None:
    values = []
    for v in dict_tuple:
        values.append(v)

    assert values == [1, 2]


@pytest.mark.parametrize("dict_tuple", TEST_CASES)
def test_convert_to_other_iterable_types(dict_tuple) -> None:
    assert list(dict_tuple) == [1, 2]
    assert tuple(dict_tuple) == (1, 2)


@pytest.mark.parametrize("dict_tuple", TEST_CASES)
def test_keys(dict_tuple) -> None:
    assert list(dict_tuple.keys()) == ["x", "y"]


@pytest.mark.parametrize("dict_tuple", TEST_CASES)
def test_get_mapping(dict_tuple) -> None:
    assert dict(dict_tuple.items()) == {"x": 1, "y": 2}


@pytest.mark.parametrize("dict_tuple", TEST_CASES)
def test_iter_over_items(dict_tuple) -> None:
    items = []
    for k, v in dict_tuple.items():
        items.append((k, v))

    assert items == [("x", 1), ("y", 2)]
