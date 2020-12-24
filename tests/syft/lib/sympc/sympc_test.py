# third party
import pytest
import torch as th

# syft absolute
import syft as sy


@pytest.mark.vendor(
    lib="sympc", python={"min_version": (3, 7)}, torch={"min_version": "1.7.0"}
)
def test_load_sympc() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()
    bob = sy.VirtualMachine()
    bob_client = bob.get_root_client()

    # third party
    from sympc.session import Session
    from sympc.tensor import MPCTensor

    sy.load_lib("sympc")

    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    y = th.Tensor([-5, 0, 1, 2, 3])
    x_secret = th.Tensor([30])
    x = MPCTensor(secret=x_secret, shape=(1,), session=session)

    assert ((x + y).reconstruct() == th.Tensor([25.0, 30.0, 31.0, 32.0, 33.0])).all()
