# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.tensor.lazy_repeat_array import lazyrepeatarray


def test_create_lazy_repeat_array() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)
    assert (lazyarray.to_numpy() == array).all()
    assert lazyarray.shape == array.shape
    assert lazyarray.size == array.size
    assert lazyarray.dtype == array.dtype


def test_create_bad_lazy_repeat_array() -> None:
    with pytest.raises(ValueError):
        _ = lazyrepeatarray(data=np.array([1]), shape=(-1, 888))

    with pytest.raises(ValueError):
        _ = lazyrepeatarray(data=np.array([1, 2, 3]), shape=(999, 888))


def test_equality() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)
    assert lazyarray == array


def test_inequality() -> None:
    array = np.array([2, 2, 2])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)
    assert lazyarray != array


def test_serde() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)

    ser = sy.serialize(lazyarray, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert lazyarray == de
    assert array == de


def test_add() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)

    assert lazyarray + 1 == array + 1
    assert lazyarray + array == array + array


def test_sub() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)

    assert lazyarray - 1 == array - 1
    assert lazyarray - array == array - array


def test_mul() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)

    assert lazyarray * 1 == array * 1
    assert lazyarray * array == array * array


def test_pow() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)

    assert lazyarray**2 == array**2


def test_astype() -> None:
    array = np.array([1, 1, 1], dtype=np.uint8)
    lazyarray = lazyrepeatarray(data=np.array([1], dtype=np.int32), shape=array.shape)
    uint8_lazyarray = lazyarray.astype(np.uint8)

    assert lazyarray == array
    assert lazyarray.dtype != array.dtype
    assert uint8_lazyarray.dtype == array.dtype
    assert uint8_lazyarray == array


def test_sum() -> None:
    array = np.array([1, 1, 1])
    lazyarray = lazyrepeatarray(data=np.array([1]), shape=array.shape)

    assert lazyarray.sum(axis=None) == array.sum(axis=None)
