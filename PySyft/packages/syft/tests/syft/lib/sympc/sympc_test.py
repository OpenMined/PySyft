# third party
import pytest
import torch as th

# syft absolute
import syft as sy

sympc = pytest.importorskip("sympc")
Session = sympc.session.Session
SessionManager = sympc.session.SessionManager
MPCTensor = sympc.tensor.MPCTensor

sy.load("sympc")


@pytest.mark.asyncio
@pytest.mark.vendor(lib="sympc")
def test_load_sympc() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()
    bob = sy.VirtualMachine()
    bob_client = bob.get_root_client()

    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)

    y = th.Tensor([-5, 0, 1, 2, 3])
    x_secret = th.Tensor([30])
    x = MPCTensor(secret=x_secret, shape=(1,), session=session)

    assert ((x + y).reconstruct() == th.Tensor([25.0, 30.0, 31.0, 32.0, 33.0])).all()
