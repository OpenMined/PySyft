import pytest
from syft.frameworks.crypten.context import run_multiworkers
import torch as th
import syft as sy
import crypten


def test_context(workers):
    # self, alice and bob
    n_workers = 3

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor([42, 53]).tag("crypten_data").send(alice)
    bob_tensor_ptr = th.tensor([101, 32]).tag("crypten_data").send(bob)

    @sy.func2plan()
    def toy_func():
        alice_tensor = crypten.load("crypten_data", 1, "alice")
        bob_tensor = crypten.load("crypten_data", 2, "bob")

        #crypt = crypten.cat([alice_tensor, bob_tensor], dim=0)
        return (alice_tensor + bob_tensor).get_plain_text().tolist()

    th.Tensor.get_plain_text = lambda x: th.tensor([1,2])

    @run_multiworkers(toy_func, [alice, bob], master_addr="127.0.0.1")
    def test_three_parties():
        pass  # pragma: no cover

    return_values = test_three_parties()
    # A toy function is ran at each party, and they should all decrypt
    # a tensor with value [42, 53, 101, 32]
    expected_value = [42, 53, 101, 32]
    for rank in range(n_workers):
        assert (
            return_values[rank] == expected_value
        ), "Crypten party with rank {} don't match expected value {} != {}".format(
            rank, return_values[rank], expected_value
        )
