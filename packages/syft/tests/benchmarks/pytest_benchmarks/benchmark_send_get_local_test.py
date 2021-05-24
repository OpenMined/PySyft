# stdlib
from typing import List

# syft absolute
import syft as sy
from syft.lib.python import List as SyList
from syft.lib.python.string import String


def send_get_string_local(data: str, duet: sy.VirtualMachine) -> None:
    syft_string = String(data)

    ptr = syft_string.send(duet, pointable=True)
    remote_data = ptr.get()

    assert data == remote_data


def send_get_list_local(data: List[str], duet: sy.VirtualMachine) -> None:
    syft_list = SyList(data)

    ptr = syft_list.send(duet, pointable=True)
    remote_data = ptr.get()

    assert data == remote_data
