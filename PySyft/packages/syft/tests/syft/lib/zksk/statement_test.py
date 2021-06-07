# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="zksk")
def test_statement_zk_proof() -> None:
    vm = sy.VirtualMachine()
    client = vm.get_root_client()

    sy.load("zksk")

    # third party
    from zksk import DLRep
    from zksk import Secret
    from zksk import utils

    num = 2
    seed = 42
    num_sy = sy.lib.python.Int(num)
    seed_sy = sy.lib.python.Int(seed)

    # Setup: Peggy and Victor agree on two group generators.
    G, H = utils.make_generators(num=num, seed=seed)
    # Setup: generate a secret randomizer.
    r = Secret(utils.get_random_num(bits=128))

    # This is Peggy's secret bit.
    top_secret_bit = 1

    # A Pedersen commitment to the secret bit.
    C = top_secret_bit * G + r.value * H

    # Peggy's definition of the proof statement, and proof generation.
    # (The first or-clause corresponds to the secret value 0, and the second to the value 1. Because
    # the real value of the bit is 1, the clause that corresponds to zero is marked as simulated.)
    stmt = DLRep(C, r * H, simulated=True) | DLRep(C - G, r * H)
    zk_proof = stmt.prove()

    # send over the network and get back
    num_ptr = num_sy.send(client)
    seed_prt = seed_sy.send(client)
    c_ptr = C.send(client)
    zk_proof_ptr = zk_proof.send(client)

    num2 = num_ptr.get().upcast()
    seed2 = seed_prt.get().upcast()
    C2 = c_ptr.get()
    zk_proof2 = zk_proof_ptr.get()

    # Setup: get the agreed group generators.
    G, H = utils.make_generators(num=num2, seed=seed2)
    # Setup: define a randomizer with an unknown value.
    r = Secret()

    stmt = DLRep(C2, r * H) | DLRep(C2 - G, r * H)
    assert stmt.verify(zk_proof2)
