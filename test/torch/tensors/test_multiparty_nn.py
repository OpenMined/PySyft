import torch
import syft as sy
from syft.frameworks.torch.mpc import securenn


def test_select_share(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    alpha_1 = torch.tensor([[0, 1]]).share(alice, bob, charlie, crypto_provider=james).child
    ones = torch.tensor([[1, 1]]).share(alice, bob, charlie, crypto_provider=james).child
    twos = torch.tensor([[2, 2]]).share(alice, bob, charlie, crypto_provider=james).child
    selected = securenn.select_share(alpha_1, ones, twos)
    assert (selected.get() == torch.tensor([[1, 2]])).all()


def test_private_compare(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    x_bits = (
        securenn.decompose(torch.tensor([3, 3]))
        .share(alice, bob, charlie, crypto_provider=james)
        .child
    )
    r = torch.tensor([1, 5]).send(alice, bob, charlie).child
    b = torch.tensor([0]).send(alice, bob, charlie).child
    compare = securenn.private_compare(x_bits, r, b)
    print("compare", compare)
    assert (compare == torch.tensor([1, 0])).all()


def test_share_convert(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    tensorA = (
        torch.tensor([10, 20, 30])
        .share(alice, bob, charlie, crypto_provider=james, field=2 ** 60)
        .child
    )
    tensorB = securenn.share_convert(tensorA)

    assert (tensorA.get() == tensorB.get()).all()


def test_msb(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    tensorA = (
        torch.tensor([-10, 0, 10])
        .share(alice, bob, charlie, crypto_provider=james, field=2 ** 60 - 1)
        .child
    )
    assert (securenn.msb(tensorA).get() == torch.tensor([1, 0, 0])).all()


def test_relu_deriv(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    tensorA = (
        torch.tensor([-10, 0, 10])
        .share(alice, bob, charlie, crypto_provider=james, field=2 ** 60)
        .child
    )
    assert (securenn.relu_deriv(tensorA).get() == torch.tensor([0, 1, 1])).all()


def test_relu(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    tensorA = (
        torch.tensor([-10, 0, 10])
        .share(alice, bob, charlie, crypto_provider=james, field=2 ** 60)
        .child
    )
    assert (securenn.relu(tensorA).get() == torch.tensor([0, 0, 10])).all()


def test_division(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    tensorA = (
        torch.tensor([[0, 10, 10, 20, 20]]).share(alice, bob, charlie, crypto_provider=james).child
    )
    tensorB = (
        torch.tensor([[1, 2, 3, 4, 5]]).share(alice, bob, charlie, crypto_provider=james).child
    )
    division = securenn.division(tensorA, tensorB)
    assert (division.get() == torch.tensor([[0, 5, 3, 5, 4]])).all()


def test_maxpool(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    tensorA = (
        torch.tensor([[0, 1, 8, 3, 4]]).share(alice, bob, charlie, crypto_provider=james).child
    )
    v, i = securenn.maxpool(tensorA)
    assert (v.get() == torch.tensor([[8]])).all()
    assert (i.get() == torch.tensor([[2]])).all()


def test_maxpool_deriv(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )
    tensorA = (
        torch.tensor([[0, 1, 8, 3]])
        .share(alice, bob, charlie, crypto_provider=james, field=2 ** 60)
        .child
    )
    deriv = securenn.maxpool_deriv(tensorA)
    assert (deriv.get() == torch.tensor([[0, 0, 1, 0]])).all()
