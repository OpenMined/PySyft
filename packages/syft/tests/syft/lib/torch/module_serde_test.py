# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.lib.python.collections import OrderedDict


@pytest.mark.slow
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
