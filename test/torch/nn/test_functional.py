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
