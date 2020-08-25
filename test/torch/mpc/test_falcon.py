from syft.frameworks.torch.mpc.falcon import Falcon

import torch
import pytest


@pytest.mark.parametrize("bit_select", [0, 1])
def test_select_share(bit_select, workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    workers = [bob, alice, james]

    x = torch.tensor([4, 5, 6])
    x_sh = x.share(*workers, protocol="falcon")

    y = torch.tensor([1, 2, 3])
    y_sh = y.share(*workers, protocol="falcon")

    b = torch.tensor(bit_select)
    b_sh = b.share(*workers, protocol="falcon", field=2)

    selected_share = Falcon.select_share(b_sh, x_sh, y_sh)
    selected_val = selected_share.reconstruct()

    if bit_select:
        assert (selected_val == y).all()
    else:
        assert (selected_val == x).all()
