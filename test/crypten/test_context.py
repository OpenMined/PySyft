import pytest
import crypten

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import syft as sy

from syft.frameworks.crypten.context import run_multiworkers
from syft.frameworks.crypten.worker_support import methods_to_add


th.set_num_threads(1)


# Define an example network
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


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
    # alice and bob
    n_workers = 2

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor([42, 53, 3, 2]).tag("crypten_data").send(alice)
    bob_tensor_ptr = th.tensor([101, 32, 29, 2]).tag("crypten_data").send(bob)

    alice.add_crypten_support()
    bob.add_crypten_support()

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    @sy.func2plan()
    def plan_func(crypten=crypten):
        alice_tensor = crypten.load("crypten_data", 0)
        bob_tensor = crypten.load("crypten_data", 1)

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


def test_context_jail(workers):
    # alice and bob
    n_workers = 2

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor([42, 53, 3, 2]).tag("crypten_data").send(alice)
    bob_tensor_ptr = th.tensor([101, 32, 29, 2]).tag("crypten_data").send(bob)

    alice.add_crypten_support()
    bob.add_crypten_support()

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    def jail_func(crypten=crypten):
        alice_tensor = crypten.load("crypten_data", 0)
        bob_tensor = crypten.load("crypten_data", 1)

        crypt = alice_tensor + bob_tensor
        result = crypt.get_plain_text()
        return result

    return_values = jail_func()

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


def test_context_jail_with_model(workers):
    dummy_input = th.empty(1, 1, 28, 28)
    pytorch_model = ExampleNet()

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor(dummy_input).tag("crypten_data").send(alice)

    alice.add_crypten_support()
    bob.add_crypten_support()

    @run_multiworkers(
        [alice, bob], master_addr="127.0.0.1", model=pytorch_model, dummy_input=dummy_input
    )
    def run_encrypted_eval():
        rank = crypten.communicator.get().get_rank()
        t = crypten.load("crypten_data", 0)

        model.encrypt()  # noqa: F821
        out = model(t)  # noqa: F821
        model.decrypt()  # noqa: F821
        out = out.get_plain_text()
        return model, out  # noqa: F821

    result = run_encrypted_eval()
    # compare out
    assert th.all(result[0][1] == result[1][1])
