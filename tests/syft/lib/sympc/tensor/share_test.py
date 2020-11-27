# third party
from sympc.tensor import ShareTensor

# syft absolute
import syft as sy


def test_share_send() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    share = ShareTensor(data=50, encoder_precision=4, encoder_base=10)

    ptr = share.send(alice_client)

    assert share == ptr.get()
