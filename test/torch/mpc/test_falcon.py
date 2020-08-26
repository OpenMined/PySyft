from syft.frameworks.torch.mpc.falcon import Falcon
from syft.frameworks.torch.tensors.interpreters.replicated_shared import ReplicatedSharingTensor

import torch
import pytest


@pytest.mark.parametrize("bit_select", [0, 1])
def test_select_share(bit_select, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]

    x = torch.tensor([0, 1, 2])
    x_sh = x.share(*workers, protocol="falcon")

    y = torch.tensor([-3, 0, 1])
    y_sh = y.share(*workers, protocol="falcon")

    b = torch.tensor(bit_select)
    b_sh = b.share(*workers, protocol="falcon", field=2)

    selected_share = Falcon.select_share(b_sh, x_sh, y_sh)
    selected_val = selected_share.reconstruct()

    if bit_select:
        assert (selected_val == y).all()
    else:
        assert (selected_val == x).all()


@pytest.mark.parametrize("beta", [0, 1])
def test_evaluate(beta, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]

    x = torch.tensor([-4, 5, 6])
    x_sh = x.share(*workers, protocol="falcon")
    field = x_sh.field
    shape = x_sh.shape

    if beta:
        beta_sh = ReplicatedSharingTensor.one_shares(workers, field=field, shape=shape)
    else:
        beta_sh = ReplicatedSharingTensor.zero_shares(workers, field=field, shape=shape)

    expected_val = (-1) ** beta * x
    evaluated_share = Falcon.evaluate(x_sh, beta_sh)
    evaluated_val = evaluated_share.reconstruct()

    assert (expected_val == evaluated_val).all()
