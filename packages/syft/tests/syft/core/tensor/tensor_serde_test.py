# stdlib
import timeit
from typing import Any

# third party
import numpy as np
from pympler.asizeof import asizeof

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.tensor import Tensor


def ishan() -> Entity:
    return Entity(name="Ishan")


def highest() -> int:
    ii32 = np.iinfo(np.int32)
    # 2147483647
    return ii32.max


def lowest() -> int:
    ii32 = np.iinfo(np.int32)
    # -2147483648
    return ii32.min


def make_bounds(data, bound: int) -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly
    generated b/w 0-1"""
    return np.ones_like(data) * bound


def make_sept(rows: int, cols: int) -> SEPT:
    upper = highest()
    lower = lowest()
    reference_data = np.ones((rows, cols), dtype=np.int32) * highest()
    return SEPT(
        child=reference_data,
        entity=ishan(),
        max_vals=make_bounds(reference_data, upper),
        min_vals=make_bounds(reference_data, lower),
    )


def size(obj: Any) -> int:
    return asizeof(obj) / (1024 * 1024)  # MBs


def test_numpy_child() -> None:
    child = np.array([1, 2, 3], dtype=np.int32)
    tensor = Tensor(child=child)

    ser = sy.serialize(tensor, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert (tensor.child == de.child).all()
    assert tensor == de


def test_sept_child() -> None:
    """We need to benchmark both the size and time to serialize and deserialize SEPTs"""
    rows = 10_000
    cols = 7
    # these times and sizes are based on the above constants and Madhavas MacBook Pro 2019
    expected_sept_mem_size = 0.8039932250976562
    expected_sept_ser_size = 0.00063323974609375
    macbook_pro_2019_ser_time = 0.0011018469999997116
    macbook_pro_2019_de_time = 0.001034114000000308

    sept = make_sept(rows=rows, cols=cols)

    start = timeit.default_timer()
    ser = sy.serialize(sept, to_bytes=True)
    end = timeit.default_timer()
    time_ser = end - start

    start = timeit.default_timer()
    de = sy.deserialize(ser, from_bytes=True)
    end = timeit.default_timer()
    time_de = end - start

    assert sept == de

    current_sept_mem_size = size(sept)
    mem_diff = (current_sept_mem_size / expected_sept_mem_size * 100) - 100

    current_sept_bytes_size = size(ser)
    bytes_diff = (current_sept_bytes_size / expected_sept_ser_size * 100) - 100

    ser_time_diff = (time_ser / macbook_pro_2019_ser_time * 100) - 100
    de_time_diff = (time_de / macbook_pro_2019_de_time * 100) - 100

    print("SEPT Stats")
    print("==========")
    print("In-memory size of SEPT", size(sept))
    print("Serialized size of SEPT", size(ser))
    print(f"Serializing {rows}x{cols} took {time_ser} secs")
    print(f"Deserializing {rows}x{cols} took {time_de} secs")

    print("Current Results")
    print("===============")
    print(f"In-memory size delta: {mem_diff}%")
    print(f"Serialized size delta: {bytes_diff}%")
    print(f"Serializing time delta: {ser_time_diff}%")
    print(f"Deserializing time delta: {de_time_diff}%")

    # we want to assert that our calculated values are smaller than the old values with
    # some tolerance
    assert (current_sept_mem_size - expected_sept_mem_size) < 1e-1
    assert (current_sept_bytes_size - expected_sept_ser_size) < 2e-3
    # TODO: make time benchmarks stable (probably can't run in parallel)
    # assert (time_ser - macbook_pro_2019_ser_time) < 2e-1
    # assert (time_de - macbook_pro_2019_de_time) < 2e-1


def test_rept_child() -> None:
    """We need to benchmark both the size and time to serialize and deserialize REPTs"""
    rows = 10_000
    cols = 7
    rept_row_count = 5

    # rows = 7
    # cols = 1
    # rept_row_count = 100_000

    # these times and sizes are based on the above constants
    # and Madhavas MacBook Pro 2019
    expected_rept_mem_size = 4.012321472167969
    expected_rept_ser_size = 0.00313568115234375
    macbook_pro_2019_ser_time = 0.002224604000000241
    macbook_pro_2019_de_time = 0.5911229659999995

    sept = make_sept(rows=rows, cols=cols)
    rept_rows = [sept.copy() for i in range(rept_row_count)]

    rept = REPT(rows=rept_rows)
    # rept.serde_concurrency = 1

    start = timeit.default_timer()
    ser = sy.serialize(rept, to_bytes=True)
    end = timeit.default_timer()
    time_ser = end - start

    start = timeit.default_timer()
    de = sy.deserialize(ser, from_bytes=True)
    end = timeit.default_timer()
    time_de = end - start

    assert rept == de

    current_rept_mem_size = size(rept)
    mem_diff = (current_rept_mem_size / expected_rept_mem_size * 100) - 100

    current_rept_bytes_size = size(ser)
    bytes_diff = (current_rept_bytes_size / expected_rept_ser_size * 100) - 100

    ser_time_diff = (time_ser / macbook_pro_2019_ser_time * 100) - 100
    de_time_diff = (time_de / macbook_pro_2019_de_time * 100) - 100

    print("REPT Stats")
    print("==========")
    print("In-memory size of REPT", size(rept))
    print("Serialized size of REPT", size(ser))
    print(f"Serializing {rept_row_count}x{rows}x{cols} took {time_ser} secs")
    print(f"Deserializing {rept_row_count}x{rows}x{cols} took {time_de} secs")

    print("Current Results")
    print("===============")
    print(f"In-memory size delta: {mem_diff}%")
    print(f"Serialized size delta: {bytes_diff}%")
    print(f"Serializing time delta: {ser_time_diff}%")
    print(f"Deserializing time delta: {de_time_diff}%")

    # we want to assert that our calculated values are smaller than the old values with
    # some tolerance
    assert (current_rept_mem_size - expected_rept_mem_size) < 1e-1
    assert (current_rept_bytes_size - expected_rept_ser_size) < 2e-2
    # TODO: make time benchmarks stable (probably can't run in parallel)
    # assert (time_ser - macbook_pro_2019_ser_time) < 2e-1
    # assert (time_de - macbook_pro_2019_de_time) < 2e-1


def time_and_size_serde(obj: Any) -> np.array:
    mem_size = size(obj)

    start = timeit.default_timer()
    ser = sy.serialize(obj, to_bytes=True)
    end = timeit.default_timer()

    time_ser = end - start
    ser_size = size(ser)

    start = timeit.default_timer()
    de = sy.deserialize(ser, from_bytes=True)
    end = timeit.default_timer()

    time_de = end - start

    assert obj == de
    return np.array([mem_size, ser_size, time_ser, time_de])


def test_big_sept_vs_rept_child() -> None:
    """Compare REPT of 50_000x1x7 vs SEPT of 50_000x7 to check overhead of REPT"""
    rows = 1
    cols = 7
    rept_row_count = 50_000

    big_sept = make_sept(rows=rows * rept_row_count, cols=cols)
    big_sept_metrics = time_and_size_serde(big_sept)

    sept = make_sept(rows=rows, cols=cols)
    rept_rows = [sept.copy() for _ in range(rept_row_count)]

    rept = REPT(rows=rept_rows)
    rept_metrics = time_and_size_serde(rept)

    # diff both metric numpy arrays and turn into a percentage of REPT
    diff_metrics = ((big_sept_metrics - rept_metrics) / rept_metrics) * 100

    print("Comparison Stats")
    print("================")
    print(f"In-memory size of big SEPT is {diff_metrics[0]}% more than REPT")
    print(f"Serialized size of big SEPT is {diff_metrics[1]}% more than REPT")
    print(f"Serializing time of big SEPT is {diff_metrics[2]}% more than REPT")
    print(f"Deserializing time of big SEPT is {diff_metrics[3]}% more than REPT")

    # REPT is should be marginally larger
    assert ((big_sept_metrics[0:2] - rept_metrics[0:2]) < 1e-3).all()
    # TODO: make time benchmarks stable (probably can't run in parallel)
    # assert ((big_sept_metrics[2:-1] - rept_metrics[2:-1]) < 1e-1).all()
