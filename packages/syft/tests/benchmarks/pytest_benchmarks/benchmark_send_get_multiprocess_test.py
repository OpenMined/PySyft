# stdlib
import socket
import time
from typing import Any
from typing import List

# syft absolute
from syft.lib.python import List as SyList
from syft.lib.python.string import String

# syft relative
from ...syft.grid.duet.process_test import SyftTestProcess

PORT = 21211


def do_send(data: Any) -> None:
    # syft absolute
    import syft as sy

    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{PORT}/")
    duet.requests.add_handler(action="accept")

    _ = data.send(duet, pointable=True)

    sy.core.common.event_loop.loop.run_forever()


def ds_get(data: Any) -> None:
    # syft absolute
    import syft as sy

    duet = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{PORT}/")

    for retry in range(10):
        if len(duet.store) != 0:
            break
        time.sleep(0.1)

    assert len(duet.store) != 0

    remote = duet.store[0].get(request_block=True, delete_obj=False)

    assert remote == data


def run_endpoints(do_runner: Any, ds_runner: Any, data: Any) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("localhost", PORT)) == 0

    do_proc = SyftTestProcess(target=do_runner, args=(data,))
    do_proc.start()

    ds_proc = SyftTestProcess(target=ds_runner, args=(data,))
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
        raise Exception(f"ds_proc is hanged for {len(data)}")


def send_get_string_multiprocess(data: String) -> None:
    run_endpoints(do_send, ds_get, String(data))


def send_get_list_multiprocess(data: List[str]) -> None:
    run_endpoints(do_send, ds_get, SyList(data))
