# stdlib
import multiprocessing as mp
from multiprocessing import Process
import time

# third party
import torch as th

# syft absolute
import syft as sy

mp.set_start_method("spawn", force=True)

# Make sure to run the local network.py server first:
#
# $ syft-network
#


def do() -> None:
    # stdlib
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio._set_running_loop(loop)
    _ = sy.logger.add(
        sink="syft_do.log",
        enqueue=True,
        colorize=False,
        diagnose=True,
        backtrace=True,
        level="TRACE",
        encoding="utf-8",
    )
    duet = sy.launch_duet(loopback=True, network_url="http://localhost:5000")
    duet.requests.add_handler(action="accept", name="gimme")
    t = th.randn(4000000)
    print("DO: Tensor sum:", t.sum())
    start = time.time()
    tp = t.send(duet, searchable=True)
    end = time.time()
    print("DO: Pointer: ", tp, "serialized in", end - start)
    print("DO: Store: ", duet.store)
    loop.run_forever()
    print("DO: DONE")


def ds() -> None:
    # stdlib
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio._set_running_loop(loop)
    _ = sy.logger.add(
        sink="syft_ds.log",
        enqueue=True,
        colorize=False,
        diagnose=True,
        backtrace=True,
        level="TRACE",
        encoding="utf-8",
    )
    duet = sy.join_duet(loopback=True, network_url="http://localhost:5000")
    time.sleep(1)
    print("DS: Store: ", duet.store)
    start = time.time()
    t = duet.store[0].get(
        request_block=True, timeout_secs=30, name="gimme", delete_obj=False
    )
    end = time.time()
    print("DS: Received in:", end - start)
    print("DS: Shape: ", t.shape)
    print("DS: Tensor sum:", t.sum())


if __name__ == "__main__":
    p1 = Process(target=do)
    p1.start()
    p2 = Process(target=ds)
    p2.start()
    p2.join()
    p1.terminate()
