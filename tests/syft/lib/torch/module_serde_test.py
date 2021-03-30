# stdlib
from typing import Any

# third party
import torch as th

# syft absolute
import syft as sy
from syft.lib.python.collections import OrderedDict


def test_linear_module() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    # Linear
    fc = th.nn.Linear(4, 2)

    # send
    fc_ptr = fc.send(alice_client)

    # remote call
    res_ptr = fc_ptr(th.rand([1, 4]))

    assert res_ptr.get().shape == th.Size((1, 2))

    # remote update state dict
    sd2 = OrderedDict(th.nn.Linear(4, 2).state_dict())
    sd2_ptr = sd2.send(alice_client)
    fc_ptr.load_state_dict(sd2_ptr)

    # get
    remote_sd2 = fc_ptr.get().state_dict()
    assert (remote_sd2["weight"] == sd2["weight"]).all()
    assert (remote_sd2["bias"] == sd2["bias"]).all()


def test_relu_module() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    # ReLU
    relu = th.nn.ReLU(inplace=True)

    # send
    relu_ptr = relu.send(alice_client)

    # remote call
    rand_data = th.rand([1, 4])
    res_ptr = relu_ptr(rand_data)
    rand_output = res_ptr.get()
    assert rand_output.shape == th.Size((1, 4))

    relu2 = relu_ptr.get()
    assert type(relu) == type(relu2)
    rand_output2 = relu2(rand_data)
    assert (rand_output2 == rand_output).all()


def test_user_module() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    # user defined model
    class M(th.nn.Module):
        def __init__(self) -> None:
            super(M, self).__init__()
            self.fc1 = th.nn.Linear(4, 2)
            self.fc2 = th.nn.Linear(2, 1)

        def forward(self, x: Any) -> Any:
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    m = M()

    # send
    m_ptr = m.send(alice_client)

    # remote update state dict
    sd = OrderedDict(M().state_dict())
    sd_ptr = sd.send(alice_client)
    m_ptr.load_state_dict(sd_ptr)

    # get
    sd2 = m_ptr.get().state_dict()

    assert (sd["fc1.weight"] == sd2["fc1.weight"]).all()
    assert (sd["fc1.bias"] == sd2["fc1.bias"]).all()
    assert (sd["fc2.weight"] == sd2["fc2.weight"]).all()
    assert (sd["fc2.bias"] == sd2["fc2.bias"]).all()
