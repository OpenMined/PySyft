# stdlib
import multiprocessing as mp
from multiprocessing import Process
import time

# third party
import torch as th

mp.set_start_method("spawn", force=True)

# Make sure to run the local network.py server first:
#
# $ syft-network
#


def do() -> None:
    # syft absolute
    import syft as sy

    _ = sy.logger.add(
        sink="syft_do.log",
        level="TRACE",
    )

    duet = sy.launch_duet(loopback=True, network_url="http://localhost:5000")
    duet.requests.add_handler(action="accept")
    t = th.randn(4000000)
    print("DO: Tensor sum:", t.sum())
    start = time.time()
    tp = t.send(duet, pointable=True)
    end = time.time()
    print("DO: Pointer: ", tp, "serialized in", end - start)
    print("DO: Store: ", duet.store)
    sy.core.common.event_loop.loop.run_forever()
    print("DO: DONE")


def ds() -> None:
    # syft absolute
    import syft as sy

    _ = sy.logger.add(
        sink="syft_ds.log",
        level="TRACE",
    )

    duet = sy.join_duet(loopback=True, network_url="http://localhost:5000")
    time.sleep(1)
    print("DS: Store: ", duet.store)
    start = time.time()
    t = duet.store[0].get(request_block=True, delete_obj=False)
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
