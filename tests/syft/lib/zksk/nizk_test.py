# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="zksk")
def test_nizk_serde() -> None:
    vm = sy.VirtualMachine()
    client = vm.get_root_client()

    # third party
    from zksk import DLRep
    from zksk import Secret
    from zksk import utils

    sy.load("zksk")

    num = 2
    seed = 42
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

    # zksk.base.NIZK
    zk_proof = stmt.prove()

    # test serde
    zk_proof_ptr = zk_proof.send(client)
    zk_proof2 = zk_proof_ptr.get()

    assert zk_proof == zk_proof2
