"""
Define benchmark tests
"""
# stdlib
import atexit
from multiprocessing import Process
from multiprocessing import set_start_method
import os
import time
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

# syft relative
from ...syft.grid.duet.signaling_server_test import run
from ..pytest_benchmarks.benchmark_send_get_local_test import send_get_list_local
from ..pytest_benchmarks.benchmark_send_get_local_test import send_get_string_local
from ..pytest_benchmarks.benchmark_send_get_multiprocess_test import (
    send_get_list_multiprocess,
)
from ..pytest_benchmarks.benchmark_send_get_multiprocess_test import (
    send_get_string_multiprocess,
)
from ..pytest_benchmarks.benchmark_send_get_multiprocess_test import PORT
from ..pytest_benchmarks.benchmarks_functions_test import list_serde
from ..pytest_benchmarks.benchmarks_functions_test import string_serde

set_start_method("spawn", force=True)

KB = 2 ** 10
MB = 2 ** 20
LIST_TEMPLATE = "a" * (10 * KB)


@pytest.fixture(scope="module")
def signaling_server() -> Process:
    print(f"creating signaling server on port {PORT}")
    grid_proc = Process(target=run, args=(PORT,))
    grid_proc.start()

    def grid_cleanup() -> None:
        print("stop signaling server")
        grid_proc.terminate()
        grid_proc.join()

    atexit.register(grid_cleanup)

    return grid_proc


@pytest.mark.benchmark
@pytest.mark.parametrize("byte_size", [10 * MB, 100 * MB])
def test_string_serde(byte_size: int, benchmark: Any) -> None:
    data = "a" * byte_size
    benchmark.pedantic(string_serde, args=(data,), rounds=3, iterations=3)


@pytest.mark.benchmark
@pytest.mark.parametrize("list_size", [10, 100, 1000])
def test_list_serde(list_size: int, benchmark: Any) -> None:
    data = [LIST_TEMPLATE] * list_size
    benchmark.pedantic(list_serde, args=(data,))


@pytest.mark.benchmark
@pytest.mark.parametrize("byte_size", [10 * KB, 100 * KB, MB, 10 * MB])
def test_duet_string_local(
    byte_size: int, benchmark: Any, root_client: sy.VirtualMachineClient
) -> None:
    data = "a" * byte_size

    benchmark.pedantic(
        send_get_string_local, args=(data, root_client), rounds=3, iterations=3
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("list_size", [10, 100, 1000])
def test_duet_list_local(
    list_size: int, benchmark: Any, root_client: sy.VirtualMachineClient
) -> None:
    data = [LIST_TEMPLATE] * list_size

    benchmark.pedantic(
        send_get_list_local, args=(data, root_client), rounds=3, iterations=3
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("byte_size", [10 * KB, 100 * KB, MB, 10 * MB])
def test_duet_string_multiprocess(
    byte_size: int, benchmark: Any, signaling_server: Process
) -> None:
    time.sleep(3)

    data = "a" * byte_size

    benchmark.pedantic(
        send_get_string_multiprocess, args=(data,), rounds=3, iterations=3
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("list_size", [10, 100, 1000])
def test_duet_list_multiprocess(
    list_size: int, benchmark: Any, signaling_server: Process
) -> None:
    time.sleep(3)

    data = [LIST_TEMPLATE] * list_size

    benchmark.pedantic(send_get_list_multiprocess, args=(data,), rounds=3, iterations=3)


@pytest.mark.skip
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "chunk_size,max_buffer",
    [
        (2 ** 14, 2 ** 18),
        (2 ** 18, 2 ** 23),
        (2 ** 18, 2 ** 24),
        (2 ** 18, 2 ** 25),
    ],
)
def test_duet_chunk_size(
    chunk_size: int, max_buffer: int, benchmark: Any, signaling_server: Process
) -> None:
    time.sleep(3)

    data = "a" * (60 * MB)

    os.environ["DC_MAX_CHUNK_SIZE"] = str(chunk_size)
    os.environ["DC_MAX_BUFSIZE"] = str(max_buffer)

    benchmark.pedantic(
        send_get_string_multiprocess, args=(data,), rounds=2, iterations=2
    )
