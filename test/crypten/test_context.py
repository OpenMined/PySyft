import pytest
import crypten

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import syft as sy

from syft.frameworks.crypten.context import run_multiworkers, run_party
from syft.frameworks.crypten.model import OnnxModel
from syft.frameworks.crypten import utils


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
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


def test_context_plan(workers):
    # alice and bob
    n_workers = 2

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor([42, 53, 3, 2]).tag("crypten_data").send(alice)
    bob_tensor_ptr = th.tensor([101, 32, 29, 2]).tag("crypten_data").send(bob)

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    @sy.func2plan()
    def plan_func(model=None, crypten=crypten):  # pragma: no cover
        alice_tensor = crypten.load("crypten_data", 0)
        bob_tensor = crypten.load("crypten_data", 1)

        crypt = alice_tensor + bob_tensor
        result = crypt.get_plain_text()
        return result

    return_values = plan_func()

    expected_value = th.tensor([143, 85, 32, 4])

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

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    def jail_func(crypten=crypten):  # pragma: no cover
        alice_tensor = crypten.load("crypten_data", 0)
        bob_tensor = crypten.load("crypten_data", 1)

        crypt = alice_tensor + bob_tensor
        result = crypt.get_plain_text()
        return result

    return_values = jail_func()

    expected_value = th.tensor([143, 85, 32, 4])

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

    @run_multiworkers(
        [alice, bob], master_addr="127.0.0.1", model=pytorch_model, dummy_input=dummy_input
    )
    def run_encrypted_eval():  # pragma: no cover
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


def test_context_jail_with_model_failures(workers):
    dummy_input = th.empty(1, 1, 28, 28)
    pytorch_model = ExampleNet()

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor(dummy_input).tag("crypten_data").send(alice)

    @run_multiworkers([alice, bob], master_addr="127.0.0.1", model=pytorch_model)
    def run_encrypted_eval():  # pragma: no cover
        rank = crypten.communicator.get().get_rank()
        t = crypten.load("crypten_data", 0)

        model.encrypt()  # noqa: F821
        out = model(t)  # noqa: F821
        model.decrypt()  # noqa: F821
        out = out.get_plain_text()
        return model, out  # noqa: F821

    with pytest.raises(ValueError):
        result = run_encrypted_eval()

    @run_multiworkers([alice, bob], master_addr="127.0.0.1", model=5)
    def run_encrypted_eval():  # pragma: no cover
        rank = crypten.communicator.get().get_rank()
        t = crypten.load("crypten_data", 0)

        model.encrypt()  # noqa: F821
        out = model(t)  # noqa: F821
        model.decrypt()  # noqa: F821
        out = out.get_plain_text()
        return model, out  # noqa: F821

    with pytest.raises(TypeError):
        result = run_encrypted_eval()

    @run_multiworkers([alice, bob], master_addr="127.0.0.1", model=pytorch_model, dummy_input=73)
    def run_encrypted_eval():  # pragma: no cover
        rank = crypten.communicator.get().get_rank()
        t = crypten.load("crypten_data", 0)

        model.encrypt()  # noqa: F821
        out = model(t)  # noqa: F821
        model.decrypt()  # noqa: F821
        out = out.get_plain_text()
        return model, out  # noqa: F821

    with pytest.raises(TypeError):
        result = run_encrypted_eval()


def test_run_party():
    expected = th.tensor(5)

    def party():  # pragma: no cover
        t = crypten.cryptensor(expected)
        return t.get_plain_text()

    t = run_party(None, party, 0, 1, "127.0.0.1", 15463, (), {})
    result = utils.unpack_values(t)
    assert result == expected


def test_duplicate_ids(workers):
    # alice and bob
    n_workers = 2

    alice = workers["alice"]
    alice2 = workers["alice"]

    @run_multiworkers([alice, alice2], master_addr="127.0.0.1")
    def jail_func(crypten=crypten):  # pragma: no cover
        pass

    with pytest.raises(RuntimeError):
        return_values = jail_func()


def test_context_plan_with_model(workers):
    dummy_input = th.empty(1, 1, 28, 28)
    pytorch_model = ExampleNet()

    alice = workers["alice"]
    bob = workers["bob"]

    alice_tensor_ptr = th.tensor(dummy_input).tag("crypten_data").send(alice)

    @run_multiworkers(
        [alice, bob], master_addr="127.0.0.1", model=pytorch_model, dummy_input=dummy_input
    )
    @sy.func2plan()
    def plan_func_model(model=None, crypten=crypten):  # noqa: F821
        t = crypten.load("crypten_data", 0)

        model.encrypt()
        out = model(t)
        model.decrypt()
        out = out.get_plain_text()
        return model, out

    result = plan_func_model()
    assert th.all(result[0][1] == result[1][1])


def test_context_plan_with_model_private(workers):
    """
    Test if we can run remote inference (using data that is not on our local
    paty) using a private model (model that is not known locally)
    """
    dummy_input = th.empty(1, 1, 28, 28)
    pytorch_model = ExampleNet()

    alice = workers["alice"]
    bob = workers["bob"]

    data_alice = th.tensor(dummy_input).tag("crypten_data").send(alice)
    model = OnnxModel.fromModel(pytorch_model, dummy_input).tag("crypten_model")

    # Model is known only by Bob and Alice and the data is at the local party
    alice_model_ptr = model.send(alice)
    bob_model_ptr = model.send(bob)

    @run_multiworkers([alice, bob], master_addr="127.0.0.1")
    @sy.func2plan()
    def plan_func_model(crypten=crypten):  # noqa: F821
        data = crypten.load("crypten_data", 0)

        # This should load the crypten model that is found at all parties
        model = crypten.load_model("crypten_model")

        model.encrypt()
        out = model(data)
        model.decrypt()
        out = out.get_plain_text()
        return out

    result = plan_func_model()
    assert th.all(result[0] == result[1])
