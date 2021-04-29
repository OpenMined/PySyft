# stdlib
from multiprocessing import set_start_method
import socket
import sys
import time
from typing import Any
from typing import Generator
from typing import List

# third party
import pytest

# syft absolute
import syft as sy

# syft relative
from ...grid.duet.process_test import SyftTestProcess

ts = pytest.importorskip("tenseal")
sy.load("tenseal")

set_start_method("spawn", force=True)


def chunks(lst: List[Any], n: int) -> Generator[Any, Any, Any]:
    """Yield successive n-sized chunks from lst.

    Args:
        lst: list of items to chunk
        n: number of items to include in each chunk

    Yields:
        single chunk of n items
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]  # noqa: E203


def do(ct_size: int, batch_size: int, signaling_server: int) -> None:
    # third party
    import numpy as np
    import tenseal as ts

    # syft absolute
    import syft as sy

    sy.load("tenseal")
    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.launch_duet(
        loopback=True, network_url=f"http://127.0.0.1:{signaling_server}/"
    )
    duet.requests.add_handler(action="accept")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1
    )
    context.global_scale = pow(2, 40)

    data = np.random.uniform(-10, 10, 100)
    enc = []
    for i in range(ct_size):
        enc.append(ts.ckks_vector(context, data))

    start = time.time()
    _ = context.send(duet, pointable=True)
    for chunk in chunks(enc, batch_size):
        _ = sy.lib.python.List(chunk).send(duet, pointable=True)
    sys.stderr.write(
        f"[{ct_size}][{batch_size}] DO sending took {time.time() - start} sec\n"
    )

    sy.core.common.event_loop.loop.run_forever()


def ds(ct_size: int, batch_size: int, signaling_server: int) -> None:
    # syft absolute
    import syft as sy

    sy.load("tenseal")
    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.join_duet(
        loopback=True, network_url=f"http://127.0.0.1:{signaling_server}/"
    )

    time.sleep(10)

    cnt = int(ct_size / batch_size)

    start = time.time()
    ctx = duet.store[0].get(request_block=True, delete_obj=False)
    for idx in range(1, cnt + 1):
        data = duet.store[idx].get(request_block=True, delete_obj=False)

        for tensor in data:
            tensor.link_context(ctx)

        assert len(data) == batch_size, len(data)

    sys.stderr.write(
        f"[{ct_size}][{batch_size}] DS get took {time.time() - start} sec\n"
    )


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_duet_ciphertext_size(signaling_server: int) -> None:
    time.sleep(3)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("localhost", signaling_server)) == 0

    for ct_size in [10, 20]:
        for batch_size in [1, 10, ct_size]:
            start = time.time()

            do_proc = SyftTestProcess(
                target=do, args=(ct_size, batch_size, signaling_server)
            )
            do_proc.start()

            ds_proc = SyftTestProcess(
                target=ds, args=(ct_size, batch_size, signaling_server)
            )
            ds_proc.start()

            ds_proc.join(120)

            do_proc.terminate()

            if do_proc.exception:
                exception, tb = do_proc.exception
                raise Exception(tb) from exception

            if ds_proc.exception:
                exception, tb = ds_proc.exception
                raise Exception(tb) from exception

            if ds_proc.is_alive():
                ds_proc.terminate()
                raise Exception(f"ds_proc is hanged for {ct_size}")

            print(
                f"test {ct_size} batch_size {batch_size} passed in {time.time() - start} seconds"
            )
