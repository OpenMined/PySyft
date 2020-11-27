# syft absolute
import syft as sy
from syft.lib.sympc.session import Session


def test_session_send() -> None:
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")

    alice_client = alice.get_client()
    bob_client = bob.get_client()

    session = Session(parties=[alice_client, bob_client])

    session.setup_mpc()

    assert len(session.session_ptr) == 2
