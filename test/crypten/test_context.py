import crypten

import torch as th
import syft as sy

from syft.frameworks.crypten.context import run_multiworkers
from syft.frameworks.crypten.worker_support import methods_to_add


def test_add_crypten_support(workers):
    alice = workers["alice"]

    for method in methods_to_add:
        assert not hasattr(
            alice, method.__name__
        ), f"Worker should not have method {method.__name__}"

    alice.add_crypten_support()

    for method in methods_to_add:
        assert hasattr(alice, method.__name__), f"Worker should have method {method.__name__}"

    alice.remove_crypten_support()

    for method in methods_to_add:
        assert not hasattr(
            alice, method.__name__
        ), f"Worker should not have method {method.__name__}"


def test_context(workers):
    # self, alice and bob
    n_workers = 3

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor([42, 53, 3, 2]).tag("crypten_data").send(alice)
    bob_tensor_ptr = th.tensor([101, 32, 29, 2]).tag("crypten_data").send(bob)

    alice.add_crypten_support()
    bob.add_crypten_support()

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    @sy.func2plan()
    def plan_func(crypten=crypten):
        alice_tensor = crypten.load("crypten_data", 1)
        bob_tensor = crypten.load("crypten_data", 2)

        crypt = alice_tensor + bob_tensor
        result = crypt.get_plain_text()
        return result

    return_values = plan_func()

    expected_value = th.tensor([143, 85, 32, 4])

    alice.remove_crypten_support()
    bob.remove_crypten_support()

    # A toy function is ran at each party, and they should all decrypt
    # a tensor with value [143, 85]
    for rank in range(n_workers):
        assert th.all(
            return_values[rank] == expected_value
        ), "Crypten party with rank {} don't match expected value {} != {}".format(
            rank, return_values[rank], expected_value
        )
