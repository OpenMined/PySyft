from syft.frameworks.torch.nn.conv import Conv2d
import syft as sy
import torch as th
import torch.nn as nn


def test_conv2d(workers):
    """
    Test the Conv2d module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """
    th.manual_seed(121)  # Truncation might not always work so we set the random seed

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    # mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    # th._C._set_mkldnn_enabled(False)

    model2 = Conv2d(1, 16, 3, bias=True)
    model = nn.Conv2d(1, 2, 3, bias=True)

    model2.weight = th.tensor(model.weight).fix_prec()
    model2.bias = th.tensor(model.bias).fix_prec()

    data = th.rand(10, 1, 28, 28)  # eg. mnist data

    out = model(data)

    out2 = model2(data.fix_prec()).float_prec()

    # Reset mkldnn to the original state
    # th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Note: absolute tolerance can be reduced by increasing precision_fractional of fix_prec()
    assert th.allclose(out, out2, atol=1e-2)

    # Test with Shared model and data
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    shared_data = data.fix_prec().share(bob, alice, crypto_provider=james)

    shared_model = model2.share(bob, alice, crypto_provider=james)

    out3 = shared_model(shared_data).get().float_prec()

    assert th.allclose(out, out3, atol=1e-2)
