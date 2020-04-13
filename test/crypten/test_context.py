import pytest
import syft as sy
from syft.frameworks.crypten.context import run_multiworkers
import torch as th
import crypten


def test_context(workers):
    # self, alice and bob
    n_workers = 3

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor([42, 53, 3, 2]).tag("crypten_data").send(alice)
    bob_tensor_ptr = th.tensor([101, 32, 29, 2]).tag("crypten_data").send(bob)

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    @sy.func2plan()
    def plan_func():
        alice_tensor = crypten.load("crypten_data", 1)
        bob_tensor = crypten.load("crypten_data", 2)

        crypt = alice_tensor + bob_tensor
        result = crypt.get_plain_text()
        return result

    return_values = plan_func()

    # A toy function is ran at each party, and they should all decrypt
    # a tensor with value [143, 85]
    expected_value = [143, 85, 32, 4]
    for rank in range(n_workers):
        assert (
            return_values[rank] == expected_value
        ), "Crypten party with rank {} don't match expected value {} != {}".format(
            rank, return_values[rank], expected_value
        )
