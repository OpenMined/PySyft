import pytest
from syft.frameworks.crypten.context import run_multiworkers
import torch as th


def test_context(workers):
    # self, alice and bob
    n_workers = 3

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor([42, 53]).tag("crypten_data").send(alice)
    bob_tensor_ptr = th.tensor([101, 32]).tag("crypten_data").send(bob)

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    def test_three_parties():
        pass  # pragma: no cover

    return_values = test_three_parties()
    print(return_values)
    # A toy function is ran at each party, and they should all decrypt
    # a tensor with value [42, 53, 101, 32]
    expected_value = [42, 53, 101, 32]
    for rank in range(n_workers):
        assert (
            return_values[rank] == expected_value
        ), "Crypten party with rank {} don't match expected value {} != {}".format(
            rank, return_values[rank], expected_value
        )
