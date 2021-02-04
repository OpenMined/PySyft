# stdlib
from multiprocessing.synchronize import Barrier
import sys
import time
from typing import List


def do_test(barriers: List[Barrier], port: int) -> None:
    # third party
    import numpy as np
    import tenseal as ts

    # syft absolute
    import syft as sy

    sy.load_lib("tenseal")
    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    duet.requests.add_handler(action="accept")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1
    )
    context.global_scale = pow(2, 40)
    enc = ts.ckks_tensor(context, np.array([1, 2, 3, 4]).reshape(2, 2))

    _ = context.send(duet, searchable=True)
    _ = enc.send(duet, searchable=True)

    sy.core.common.event_loop.loop.run_forever()


def ds_test(barriers: List[Barrier], port: int) -> None:
    # syft absolute
    import syft as sy

    sy.load_lib("tenseal")
    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")

    time.sleep(2)

    ctx = duet.store[0].get(request_block=True, delete_obj=False)
    data = duet.store[1].get(request_block=True, delete_obj=False)
    data.link_context(ctx)

    assert data.shape == [2, 2], data.shape


test_scenario_tenseal_sanity = ("test_scenario_tenseal_sanity", do_test, ds_test)
