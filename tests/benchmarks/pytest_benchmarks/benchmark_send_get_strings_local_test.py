# syft absolute
import syft as sy
from syft.lib.python.string import String


def send_get_string_local(data: str, duet: sy.VirtualMachine) -> None:
    syft_string = String(data)

    ptr = syft_string.send(duet, searchable=True)
    remote_data = ptr.get()

    assert data == remote_data
