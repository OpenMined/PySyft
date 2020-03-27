import torch
import syft.frameworks.torch.nn.functional as F


def test_torch_nn_functional_dropout(workers):

    # Test for fixed precision tensor
    a = torch.rand((20, 20))
    x = a.fix_prec()

    train_output = F.dropout(x, p=0.5, training=True, inplace=False)
    assert (train_output.float_prec() == 0).sum() > 0

    # training = False, should return the same input
    test_output = F.dropout(x, p=0.5, training=False, inplace=False)
    assert ((test_output == x).float_prec() == 1).all()

    # For AST wrapped under Fixed Precision
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = a.fix_prec().share(alice, bob, crypto_provider=james)

    train_output = F.dropout(x, p=0.5, training=True, inplace=False)
    assert (train_output.get().float_prec() == 0).sum() > 0

    # training = False, should return the same input
    test_output = F.dropout(x, p=0.5, training=False, inplace=False)
    assert ((test_output == x).get().float_prec() == 1).all()


def test_torch_nn_functional_conv2d(workers):
    # Test with FixedPrecision tensors
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    im = torch.Tensor(
        [
            [
                [[0.5, 1.0, 2.0], [3.5, 4.0, 5.0], [6.0, 7.5, 8.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.5, 15.0], [16.0, 17.5, 18.0]],
            ]
        ]
    )
    w = torch.Tensor(
        [
            [[[0.0, 3.0], [1.5, 1.0]], [[2.0, 2.0], [2.5, 2.0]]],
            [[[-0.5, -1.0], [-2.0, -1.5]], [[0.0, 0.0], [0.0, 0.5]]],
        ]
    )
    bias = torch.Tensor([-1.3, 15.0])

    im_fp = im.fix_prec()
    w_fp = w.fix_prec()
    bias_fp = bias.fix_prec()

    res0 = F.conv2d(im_fp, w_fp, bias=bias_fp, stride=1).float_prec()
    res1 = F.conv2d(
        im_fp, w_fp[:, 0:1].contiguous(), bias=bias_fp, stride=2, padding=3, dilation=2, groups=2
    ).float_prec()

    expected0 = torch.conv2d(im, w, bias=bias, stride=1)
    expected1 = torch.conv2d(
        im, w[:, 0:1].contiguous(), bias=bias, stride=2, padding=3, dilation=2, groups=2
    )

    assert (res0 == expected0).all()
    assert (res1 == expected1).all()

    # Test with AdditiveSharing tensors (Wrapper)>FixedPrecision>AdditiveShared
    im_shared = im.fix_prec().share(bob, alice, crypto_provider=james)
    w_shared = w.fix_prec().share(bob, alice, crypto_provider=james)
    bias_shared = bias.fix_prec().share(bob, alice, crypto_provider=james)

    res0 = F.conv2d(im_shared, w_shared, bias=bias_shared, stride=1).get().float_precision()
    res1 = (
        F.conv2d(
            im_shared,
            w_shared[:, 0:1].contiguous(),
            bias=bias_shared,
            stride=2,
            padding=3,
            dilation=2,
            groups=2,
        )
        .get()
        .float_precision()
    )

    expected0 = torch.conv2d(im, w, bias=bias, stride=1)
    expected1 = torch.conv2d(
        im, w[:, 0:1].contiguous(), bias=bias, stride=2, padding=3, dilation=2, groups=2
    )

    assert (res0 == expected0).all()
    assert (res1 == expected1).all()
