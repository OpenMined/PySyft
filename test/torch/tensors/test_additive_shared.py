import copy
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

import syft
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor


def test_wrap(workers):
    """
    Test the .on() wrap functionality for AdditiveSharingTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = AdditiveSharingTensor().on(x_tensor)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, AdditiveSharingTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test___str__(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x_sh = torch.tensor([[3, 4]]).share(alice, bob, crypto_provider=james)
    assert isinstance(x_sh.__str__(), str)


def test_share_get(workers):

    t = torch.tensor([1, 2, 3])
    x = t.share(workers["bob"], workers["alice"], workers["james"])

    x = x.get()

    assert (x == t).all()


def test___bool__(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x_sh = torch.tensor([[3, 4]]).share(alice, bob, crypto_provider=james)

    with pytest.raises(ValueError):
        if x_sh:  # pragma: no cover
            pass


def test_share_inplace_consistency(workers):
    """Verify that share_ produces the same output then share"""
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x1 = torch.tensor([-1.0])
    x1.fix_precision_().share_(alice, bob, crypto_provider=james)

    x2 = torch.tensor([-1.0])
    x2_sh = x2.fix_precision().share(alice, bob, crypto_provider=james)

    assert (x1 == x2_sh).get().float_prec()


def test_clone(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([1.2]).fix_precision().share(alice, bob, crypto_provider=james)
    original_props = (
        x.id,
        x.owner.id,
        x.child.id,
        x.child.owner.id,
        x.child.child.id,
        x.child.child.owner.id,
    )
    xc = x.clone()
    cloned_props = (
        xc.id,
        xc.owner.id,
        xc.child.id,
        xc.child.owner.id,
        xc.child.child.id,
        xc.child.child.owner.id,
    )
    assert original_props == cloned_props


def test_virtual_get(workers):
    t = torch.tensor([1, 2, 3])
    x = t.share(workers["bob"], workers["alice"], workers["james"])

    x = x.child.virtual_get()

    assert (x == t).all()


def test_non_client_registration(hook, workers):
    hook.local_worker.is_client_worker = False
    bob, alice, james = workers["bob"], workers["alice"], workers["james"]
    x = torch.tensor([-1.0])
    x_sh = x.fix_precision().share(alice, bob, crypto_provider=james)

    assert x_sh.id in hook.local_worker.object_store._objects
    assert (x_sh == hook.local_worker.get_obj(x_sh.id)).get().float_prec()
    hook.local_worker.is_client_worker = True


def test_autograd_kwarg(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    t = torch.tensor([1, 2, 3])
    x = t.share(alice, bob, crypto_provider=james, requires_grad=True)

    assert isinstance(x.child, syft.AutogradTensor)


def test_send_get(workers):
    # For int dtype
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x_sh = torch.tensor([[3, 4]]).fix_prec(dtype="int").share(alice, bob, crypto_provider=james)

    alice_t_id = x_sh.child.child.child["alice"].id_at_location
    assert alice_t_id in alice.object_store._objects

    ptr_x = x_sh.send(james)
    ptr_x_id_at_location = ptr_x.id_at_location
    assert ptr_x_id_at_location in james.object_store._objects
    assert alice_t_id in alice.object_store._objects

    x_sh_back = ptr_x.get()
    assert ptr_x_id_at_location not in james.object_store._objects
    assert alice_t_id in alice.object_store._objects

    x = x_sh_back.get()
    assert alice_t_id not in alice.object_store._objects

    # For long dtype
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x_sh = torch.tensor([[3, 4]]).fix_prec().share(alice, bob, crypto_provider=james)

    alice_t_id = x_sh.child.child.child["alice"].id_at_location
    assert alice_t_id in alice.object_store._objects

    ptr_x = x_sh.send(james)
    ptr_x_id_at_location = ptr_x.id_at_location
    assert ptr_x_id_at_location in james.object_store._objects
    assert alice_t_id in alice.object_store._objects

    x_sh_back = ptr_x.get()
    assert ptr_x_id_at_location not in james.object_store._objects
    assert alice_t_id in alice.object_store._objects

    x = x_sh_back.get()
    assert alice_t_id not in alice.object_store._objects


def test_add(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    # 2 workers
    t = torch.tensor([1, 2, 3])
    x = torch.tensor([1, 2, 3]).share(bob, alice)

    y = (x + x).get()

    # 3 workers
    assert (y == (t + t)).all()

    t = torch.tensor([1, 2, 3])
    x = torch.tensor([1, 2, 3]).share(bob, alice, james)

    y = (x + x).get()

    # negative numbers
    assert (y == (t + t)).all()

    t = torch.tensor([1, -2, 3])
    x = torch.tensor([1, -2, 3]).share(bob, alice, james)

    y = (x + x).get()

    assert (y == (t + t)).all()

    # with fixed precisions
    t = torch.tensor([1.0, -2, 3])
    x = torch.tensor([1.0, -2, 3]).fix_prec().share(bob, alice, james)

    y = (x + x).get().float_prec()

    assert (y == (t + t)).all()

    # with FPT>torch.tensor
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    y = t.fix_prec()

    z = (x + y).get().float_prec()

    assert (z == (t + t)).all()

    z = (y + x).get().float_prec()

    assert (z == (t + t)).all()

    # with constant integer
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, crypto_provider=james)
    c = 4

    z = (x + c).get().float_prec()
    assert (z == (t + c)).all()

    z = (c + x).get().float_prec()
    assert (z == (c + t)).all()

    # with constant float
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, crypto_provider=james)
    c = 4.2

    z = (x + c).get().float_prec()
    assert ((z - (t + c)) < 10e-3).all()

    z = (c + x).get().float_prec()
    assert ((z - (c + t)) < 10e-3).all()

    # with dtype int
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec(dtype="int").share(alice, bob, crypto_provider=james)
    y = x + x
    assert (y.get().float_prec() == torch.tensor([2.0, -4.0, 6.0])).all()


def test_sub(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    # 3 workers
    t = torch.tensor([1, 2, 3])
    x = torch.tensor([1, 2, 3]).share(bob, alice, james)

    y = (x - x).get()

    assert (y == (t - t)).all()

    # negative numbers
    t = torch.tensor([1, -2, 3])
    x = torch.tensor([1, -2, 3]).share(bob, alice, james)

    y = (x - x).get()

    assert (y == (t - t)).all()

    # with fixed precision
    t = torch.tensor([1.0, -2, 3])
    x = torch.tensor([1.0, -2, 3]).fix_prec().share(bob, alice, james)

    y = (x - x).get().float_prec()

    assert (y == (t - t)).all()

    # with FPT>torch.tensor
    t = torch.tensor([1.0, -2.0, 3.0])
    u = torch.tensor([4.0, 3.0, 2.0])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    y = u.fix_prec()

    z = (x - y).get().float_prec()

    assert (z == (t - u)).all()

    z = (y - x).get().float_prec()

    assert (z == (u - t)).all()

    # with constant integer
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, crypto_provider=james)
    c = 4

    z = (x - c).get().float_prec()
    assert (z == (t - c)).all()

    z = (c - x).get().float_prec()
    assert (z == (c - t)).all()

    # with constant float
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(alice, bob, crypto_provider=james)
    c = 4.2

    z = (x - c).get().float_prec()
    assert ((z - (t - c)) < 10e-3).all()

    z = (c - x).get().float_prec()
    assert ((z - (c - t)) < 10e-3).all()

    # with dtype int
    t = torch.tensor([1.0, -2.0, 3.0])
    u = torch.tensor([4.0, 3.0, 2.0])
    x = t.fix_prec(dtype="int").share(alice, bob, crypto_provider=james)
    y = u.fix_prec(dtype="int").share(alice, bob, crypto_provider=james)
    z = y - x
    assert (z.get().float_prec() == torch.tensor([3.0, 5.0, -1.0])).all()


def test_mul(workers):
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed
    bob, alice, james, charlie = (
        workers["bob"],
        workers["alice"],
        workers["james"],
        workers["charlie"],
    )

    # 2 workers
    t = torch.tensor([1, 2, 3, 4])
    x = t.share(bob, alice, crypto_provider=james)
    y = x * x
    assert (y.get() == (t * t)).all()

    # 3 workers
    t = torch.tensor([1, 2, 3, 4])
    x = t.share(bob, alice, charlie, crypto_provider=james)
    y = x * x
    assert (y.get() == (t * t)).all()

    # with fixed precision
    x = torch.tensor([1, -2, -3, 4.0]).fix_prec().share(bob, alice, crypto_provider=james)
    y = torch.tensor([-1, 2, -3, 4.0]).fix_prec().share(bob, alice, crypto_provider=james)
    y = (x * y).get().float_prec()

    assert (y == torch.tensor([-1, -4, 9, 16.0])).all()

    # with non-default fixed precision
    t = torch.tensor([1, 2, 3, 4.0])
    x = t.fix_prec(precision_fractional=2).share(bob, alice, crypto_provider=james)
    y = (x * x).get().float_prec()

    assert (y == (t * t)).all()

    # with FPT>torch.tensor
    t = torch.tensor([1.0, -2.0, 3.0])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    y = t.fix_prec()

    z = (x * y).get().float_prec()

    assert (z == (t * t)).all()

    # with dtype int
    x = (
        torch.tensor([1, -2, -3, 4.0])
        .fix_prec(dtype="int")
        .share(bob, alice, crypto_provider=james)
    )
    y = (
        torch.tensor([-1, 2, -3, 4.0])
        .fix_prec(dtype="int")
        .share(bob, alice, crypto_provider=james)
    )
    z = x * y
    assert (z.get().float_prec() == torch.tensor([-1, -4, 9, 16.0])).all()


def test_public_mul(workers):
    bob, alice, james, charlie = (
        workers["bob"],
        workers["alice"],
        workers["james"],
        workers["charlie"],
    )

    t = torch.tensor([-3.1, 1.0])
    x = t.fix_prec().share(alice, bob, crypto_provider=james)
    y = 1
    z = (x * y).get().float_prec()
    assert (z == (t * y)).all()

    # 3 workers
    t = torch.tensor([-3.1, 1.0])
    x = t.fix_prec().share(alice, bob, charlie, crypto_provider=james)
    y = 1
    z = (x * y).get().float_prec()
    assert (z == (t * y)).all()

    t = torch.tensor([-3.1, 1.0])
    x = t.fix_prec().share(alice, bob, crypto_provider=james)
    y = 0
    z = (x * y).get().float_prec()
    assert (z == (t * y)).all()

    t_x = torch.tensor([-3.1, 1])
    t_y = torch.tensor([1.0])
    x = t_x.fix_prec().share(alice, bob, crypto_provider=james)
    y = t_y.fix_prec()
    z = x * y
    z = z.get().float_prec()
    assert (z == t_x * t_y).all()

    t_x = torch.tensor([-3.1, 1])
    t_y = torch.tensor([0.0])
    x = t_x.fix_prec().share(alice, bob, crypto_provider=james)
    y = t_y.fix_prec()
    z = x * y
    z = z.get().float_prec()
    assert (z == t_x * t_y).all()

    t_x = torch.tensor([-3.1, 1])
    t_y = torch.tensor([0.0, 2.1])
    x = t_x.fix_prec().share(alice, bob, crypto_provider=james)
    y = t_y.fix_prec()
    z = x * y
    z = z.get().float_prec()
    assert (z == t_x * t_y).all()

    # with dtype int
    t_x = torch.tensor([-3.1, 1])
    t_y = torch.tensor([0.0, 2.1])
    x = t_x.fix_prec(dtype="int").share(alice, bob, crypto_provider=james)
    y = t_y.fix_prec(dtype="int")
    z = x * y
    z = z.get().float_prec()
    assert (z == t_x * t_y).all()


def test_div(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    # With scalar
    t = torch.tensor([[9.0, 12.0], [3.3, 0.0]])
    x = t.fix_prec(dtype="long").share(bob, alice, crypto_provider=james)
    y = (x / 3).get().float_prec()

    assert (y == torch.tensor([[3.0, 4.0], [1.1, 0.0]])).all()

    # With another encrypted tensor of same shape
    t1 = torch.tensor([[25, 9], [10, 30]])
    t2 = torch.tensor([[5, 12], [2, 7]])
    x1 = t1.fix_prec(dtype="long").share(bob, alice, crypto_provider=james)
    x2 = t2.fix_prec(dtype="long").share(bob, alice, crypto_provider=james)

    y = (x1 / x2).get().float_prec()
    assert (y == torch.tensor([[5.0, 0.75], [5.0, 4.285]])).all()

    # With another encrypted single value
    t1 = torch.tensor([[25.0, 9], [10, 30]])
    t2 = torch.tensor([5.0])
    x1 = t1.fix_prec(dtype="long").share(bob, alice, crypto_provider=james)
    x2 = t2.fix_prec(dtype="long").share(bob, alice, crypto_provider=james)

    y = (x1 / x2).get().float_prec()
    assert (y == torch.tensor([[5.0, 1.8], [2.0, 6.0]])).all()


def test_pow(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    m = torch.tensor([[1, 2], [3, 4.0]])
    x = m.fix_prec().share(bob, alice, crypto_provider=james)
    y = (x ** 3).get().float_prec()

    assert (y == (m ** 3)).all()


def test_operate_with_integer_constants(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    x = torch.tensor([2.0])
    x_sh = x.fix_precision().share(alice, bob, crypto_provider=james)

    r_sh = x_sh + 10
    assert r_sh.get().float_prec() == x + 10

    r_sh = x_sh - 7
    assert r_sh.get().float_prec() == x - 7

    r_sh = x_sh * 2
    assert r_sh.get().float_prec() == x * 2

    r_sh = x_sh / 2
    assert r_sh.get().float_prec() == x / 2


def test_stack(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    t = torch.tensor([1.3, 2])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    res = torch.stack([x, x]).get().float_prec()

    expected = torch.tensor([[1.3000, 2.0000], [1.3000, 2.0000]])

    assert (res == expected).all()


def test_cat(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x = t.share(bob, alice, crypto_provider=james)

    res0 = torch.cat([x, x], dim=0).get()
    res1 = torch.cat([x, x], dim=1).get()

    expected0 = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
    expected1 = torch.tensor([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])

    assert (res0 == expected0).all()
    assert (res1 == expected1).all()

    # Test when using more tensors
    res2 = torch.cat([x, x, x], dim=1).get()
    expected2 = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6]])

    assert (res2 == expected2).all()


def test_chunk(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    x = t.share(bob, alice, crypto_provider=james)

    res0 = torch.chunk(x, 2, dim=0)
    res1 = torch.chunk(x, 2, dim=1)

    expected0 = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]])]
    expected1 = [torch.tensor([[1, 2], [5, 6]]), torch.tensor([[3, 4], [7, 8]])]

    assert all([(res0[i].get() == expected0[i]).all() for i in range(2)])
    assert all([(res1[i].get() == expected1[i]).all() for i in range(2)])


def test_roll(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x = t.share(bob, alice, crypto_provider=james)

    res1 = torch.roll(x, 2)
    res2 = torch.roll(x, 2, dims=1)
    res3 = torch.roll(x, (1, 2), dims=(0, 1))

    assert (res1.get() == torch.roll(t, 2)).all()
    assert (res2.get() == torch.roll(t, 2, dims=1)).all()
    assert (res3.get() == torch.roll(t, (1, 2), dims=(0, 1))).all()

    # With MultiPointerTensor
    shifts = torch.tensor(1).send(alice, bob)
    res = torch.roll(x, shifts)

    shifts1 = torch.tensor(1).send(alice, bob)
    shifts2 = torch.tensor(2).send(alice, bob)
    res2 = torch.roll(x, (shifts1, shifts2), dims=(0, 1))

    assert (res.get() == torch.roll(t, 1)).all()
    assert (res2.get() == torch.roll(t, (1, 2), dims=(0, 1))).all()


def test_matmul(workers):
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    m = torch.tensor([[1, 2], [3, 4.0]])
    x = m.fix_prec().share(bob, alice, crypto_provider=james)
    y = (x @ x).get().float_prec()

    assert (y == (m @ m)).all()

    # with FPT>torch.tensor
    m = torch.tensor([[1, 2], [3, 4.0]])
    x = m.fix_prec().share(bob, alice, crypto_provider=james)
    y = m.fix_prec()

    z = (x @ y).get().float_prec()

    assert (z == (m @ m)).all()

    z = (y @ x).get().float_prec()

    assert (z == (m @ m)).all()


def test_mm(workers):
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    t = torch.tensor([[1, 2], [3, 4.0]])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)

    # Using the method
    y = (x.mm(x)).get().float_prec()
    assert (y == (t.mm(t))).all()

    # Using the function
    y = (torch.mm(x, x)).get().float_prec()
    assert (y == (torch.mm(t, t))).all()

    # with FPT>torch.tensor
    t = torch.tensor([[1, 2], [3, 4.0]])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    y = t.fix_prec()

    # Using the method
    z = (x.mm(y)).get().float_prec()
    assert (z == (t.mm(t))).all()

    # Using the function
    z = (torch.mm(x, y)).get().float_prec()
    assert (z == (torch.mm(t, t))).all()

    # Using the method
    z = (y.mm(x)).get().float_prec()
    assert (z == (t.mm(t))).all()

    # Using the function
    z = (torch.mm(y, x)).get().float_prec()
    assert (z == (torch.mm(t, t))).all()


def test_fixed_precision_and_sharing(workers):

    bob, alice = (workers["bob"], workers["alice"])

    t = torch.tensor([1, 2, 3, 4.0])
    x = t.fix_prec().share(bob, alice)
    out = x.get().float_prec()

    assert (out == t).all()

    x = t.fix_prec().share(bob, alice)

    y = x + x

    y = y.get().float_prec()
    assert (y == (t + t)).all()


def test_fixed_precision_and_sharing_on_pointer(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    t = torch.tensor([1, 2, 3, 4.0])
    ptr = t.send(james)

    x = ptr.fix_prec().share(bob, alice)

    y = x + x

    y = y.get().get().float_prec()
    assert (y == (t + t)).all()


def test_pointer_on_fixed_precision_and_sharing(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    t = torch.tensor([1, 2, 3, 4.0])

    x = t.fix_prec().share(bob, alice)
    x = x.send(james)

    y = x + x

    y = y.get().get().float_prec()
    assert (y == (t + t)).all()


def test_get_item(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    x = torch.tensor([[3.1, 4.3]]).fix_prec().share(alice, bob, crypto_provider=james)
    idx = torch.tensor([0]).send(alice, bob)

    # Operate directly AST[MPT]
    assert x.child.child[:, idx.child].get() == torch.tensor([[3100]])

    # With usual wrappers and FPT
    x = torch.tensor([[3, 4]]).share(alice, bob, crypto_provider=james)
    idx = torch.tensor([0]).send(alice, bob)
    assert x[:, idx].get() == torch.tensor([[3]])


@pytest.mark.parametrize("protocol", ["snn", "fss"])
def test_eq(workers, protocol):
    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    if protocol == "fss":
        for worker in workers.values():
            syft.frameworks.torch.mpc.fss.initialize_crypto_plans(worker)
        me.crypto_store.provide_primitives(["fss_eq"], [alice, bob], n_instances=6)

    args = (alice, bob)
    kwargs = dict(protocol=protocol, crypto_provider=crypto_provider)

    x = torch.tensor([3.1]).fix_prec().share(*args, **kwargs)
    y = torch.tensor([3.1]).fix_prec().share(*args, **kwargs)

    assert (x == y).get().float_prec()

    x = torch.tensor([3.1]).fix_prec().share(*args, **kwargs)
    y = torch.tensor([2.1]).fix_prec().share(*args, **kwargs)

    assert not (x == y).get().float_prec()

    x = torch.tensor([-3.1]).fix_prec().share(*args, **kwargs)
    y = torch.tensor([-3.1]).fix_prec().share(*args, **kwargs)

    assert (x == y).get().float_prec()


@pytest.mark.parametrize("protocol", ["snn", "fss"])
def test_comp(workers, protocol):
    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    if protocol == "fss":
        for worker in workers.values():
            syft.frameworks.torch.mpc.fss.initialize_crypto_plans(worker)
        me.crypto_store.provide_primitives(
            ["xor_add_couple", "fss_eq", "fss_comp"], [alice, bob], n_instances=50
        )

    args = (alice, bob)
    kwargs = dict(protocol=protocol, crypto_provider=crypto_provider)

    x = torch.tensor([3.1]).fix_prec().share(*args, **kwargs)
    y = torch.tensor([3.1]).fix_prec().share(*args, **kwargs)

    assert (x >= y).get().float_prec()
    assert (x <= y).get().float_prec()
    assert not (x > y).get().float_prec()
    assert not (x < y).get().float_prec()

    x = torch.tensor([-3.1]).fix_prec().share(*args, **kwargs)
    y = torch.tensor([-3.1]).fix_prec().share(*args, **kwargs)

    assert (x >= y).get().float_prec()
    assert (x <= y).get().float_prec()
    assert not (x > y).get().float_prec()
    assert not (x < y).get().float_prec()

    x = torch.tensor([3.1]).fix_prec().share(*args, **kwargs)
    y = torch.tensor([2.1]).fix_prec().share(*args, **kwargs)

    assert (x >= y).get().float_prec()
    assert not (x <= y).get().float_prec()
    assert (x > y).get().float_prec()
    assert not (x < y).get().float_prec()

    t1 = torch.tensor([-2.1, 1.8])
    t2 = torch.tensor([-3.1, 0.3])
    x = t1.fix_prec().share(*args, **kwargs)
    y = t2.fix_prec().share(*args, **kwargs)

    assert ((x >= y).get().float_prec() == (t1 >= t2)).all()
    assert ((x <= y).get().float_prec() == (t1 <= t2)).all()
    assert ((x > y).get().float_prec() == (t1 > t2)).all()
    assert ((x < y).get().float_prec() == (t1 < t2)).all()

    t1 = torch.tensor([[-2.1, 1.8], [-1.1, -0.7]])
    t2 = torch.tensor([[-3.1, 0.3], [-1.1, 0.3]])
    x = t1.fix_prec().share(*args, **kwargs)
    y = t2.fix_prec().share(*args, **kwargs)

    assert ((x >= y).get().float_prec() == (t1 >= t2)).all()
    assert ((x <= y).get().float_prec() == (t1 <= t2)).all()
    assert ((x > y).get().float_prec() == (t1 > t2)).all()
    assert ((x < y).get().float_prec() == (t1 < t2)).all()


@pytest.mark.parametrize("protocol", ["snn", "fss"])
def test_max(workers, protocol):
    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    if protocol == "fss":
        for worker in workers.values():
            syft.frameworks.torch.mpc.fss.initialize_crypto_plans(worker)
        me.crypto_store.provide_primitives(
            ["xor_add_couple", "fss_eq", "fss_comp"], [alice, bob], n_instances=16
        )

    args = (alice, bob)
    kwargs = dict(protocol=protocol, crypto_provider=crypto_provider)

    t = torch.tensor([3, 1.0, 2])
    x = t.fix_prec().share(*args, **kwargs)
    max_value = x.max().get().float_prec()
    assert max_value == torch.tensor([3.0])

    t = torch.tensor([3, 4.0])
    x = t.fix_prec().share(*args, **kwargs)
    max_value = x.max().get().float_prec()
    assert max_value == torch.tensor([4.0])

    t = torch.tensor([3, 4.0, 5, 2])
    x = t.fix_prec().share(*args, **kwargs)
    max_value = x.max().get().float_prec()
    assert max_value == torch.tensor([5.0])


@pytest.mark.parametrize("protocol", ["snn", "fss"])
def test_argmax(workers, protocol):
    me, alice, bob, crypto_provider = (
        workers["me"],
        workers["alice"],
        workers["bob"],
        workers["james"],
    )

    if protocol == "fss":
        for worker in workers.values():
            syft.frameworks.torch.mpc.fss.initialize_crypto_plans(worker)
        me.crypto_store.provide_primitives(
            ["xor_add_couple", "fss_eq", "fss_comp"], [alice, bob], n_instances=32
        )

    args = (alice, bob)
    kwargs = dict(protocol=protocol, crypto_provider=crypto_provider)

    t = torch.tensor([3, 1.0, 2])
    x = t.fix_prec().share(*args, **kwargs)
    idx = x.argmax().get().float_prec()
    assert idx == torch.tensor([0.0])

    t = torch.tensor([3, 4.0])
    x = t.fix_prec().share(*args, **kwargs)
    idx = x.argmax().get().float_prec()
    assert idx == torch.tensor([1.0])

    t = torch.tensor([3, 4.0, 5, 2])
    x = t.fix_prec().share(*args, **kwargs)
    idx = x.argmax().get().float_prec()
    assert idx == torch.tensor([2.0])

    # no dim=
    t = torch.tensor([[1, 2.0, 4], [3, 9.0, 2.0]])
    x = t.fix_prec().share(*args, **kwargs)
    ids = x.argmax().get().float_prec()
    assert ids.long() == torch.argmax(t)

    # dim=1
    t = torch.tensor([[1, 2.0, 4], [3, 1.0, 2.0]])
    x = t.fix_prec().share(*args, **kwargs)
    ids = x.argmax(dim=1).get().float_prec()
    assert (ids.long() == torch.argmax(t, dim=1)).all()


def test_mod(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    t = torch.tensor([21]).share(bob, alice, crypto_provider=james)
    assert t.child.mod(8).get() % 8 == torch.tensor([5])
    assert t.child.mod(-8).get() % -8 == torch.tensor([-3])

    t = torch.tensor([-21]).share(bob, alice, crypto_provider=james)
    assert t.child.mod(8).get() % 8 == torch.tensor([3])
    assert t.child.mod(-8).get() % -8 == torch.tensor([-5])

    assert (t.child % 8).get() % 8 == torch.tensor([3])


def test_torch_sum(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    t = torch.tensor([[1, 2, 4], [8, 5, 6]])
    x = t.share(alice, bob, crypto_provider=james)

    s = torch.sum(x).get()
    s_dim = torch.sum(x, 0).get()
    s_dim2 = torch.sum(x, (0, 1)).get()
    s_keepdim = torch.sum(x, 1, keepdim=True).get()

    assert (s == torch.sum(t)).all()
    assert (s_dim == torch.sum(t, 0)).all()
    assert (s_dim2 == torch.sum(t, (0, 1))).all()
    assert (s_keepdim == torch.sum(t, 1, keepdim=True)).all()


def test_torch_mean(workers):
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    base = 10
    prec_frac = 4

    t = torch.tensor([[1.0, 2.5], [8.0, 5.5]])
    x = t.fix_prec(base=base, precision_fractional=prec_frac).share(
        alice, bob, crypto_provider=james
    )

    s = torch.mean(x).get().float_prec()
    s_dim = torch.mean(x, 0).get().float_prec()
    s_dim2 = torch.mean(x, (0, 1)).get().float_prec()
    s_keepdim = torch.mean(x, 1, keepdim=True).get().float_prec()

    assert (s == torch.tensor(4.25)).all()
    assert (s_dim == torch.tensor([4.5, 4.0])).all()
    assert (s_dim2 == torch.tensor(4.25)).all()
    assert (s_keepdim == torch.tensor([[1.75], [6.75]])).all()


def test_torch_dot(workers):
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).fix_prec().share(alice, bob, crypto_provider=james)
    y = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0]).fix_prec().share(alice, bob, crypto_provider=james)

    assert torch.dot(x, y).get().float_prec() == 45


def test_unbind(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x = torch.tensor([21, 17]).share(bob, alice, crypto_provider=james).child

    x0, x1 = torch.unbind(x)

    assert x0.get() == torch.tensor(21)
    assert x1.get() == torch.tensor(17)


def test_handle_func_command(workers):
    """
    Just to show that handle_func_command works
    Even if torch.abs should be hooked to return a correct value
    """
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    t = torch.tensor([-21]).share(bob, alice, crypto_provider=james).child
    _ = torch.abs(t).get()


def test_init_with_no_crypto_provider(workers):
    alice, bob = workers["alice"], workers["bob"]

    x = torch.tensor([21, 17]).share(bob, alice).child

    assert x.crypto_provider.id == syft.hook.local_worker.id


def test_zero_refresh(workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    t = torch.tensor([2.2, -1.0])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)

    x_sh = x.child.child
    assert (x_sh.zero().get() == torch.zeros(*t.shape).long()).all()

    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    x_copy = t.fix_prec().share(bob, alice, crypto_provider=james)
    x_r = x.refresh()

    assert (x_r.get().float_prec() == x_copy.get().float_prec()).all()

    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    x_r = x.refresh()

    assert ((x_r / 2).get().float_prec() == t / 2).all()


def test_correct_tag_and_description_after_send(workers):
    bob, alice, james, me = (workers["bob"], workers["alice"], workers["james"], workers["me"])

    x = torch.tensor([1, 2, 3]).share(alice, bob, james)
    x.tags = ["tag_additive_test1", "tag_additive_test2"]
    x.description = "description_additive_test"

    pointer_x = x.send(alice)

    assert me.request_search("tag_additive_test1", location=alice)
    assert me.request_search("tag_additive_test2", location=alice)


def test_dtype(workers):
    alice, bob, james, me = (workers["bob"], workers["alice"], workers["james"], workers["me"])
    # Without fix_prec
    x = torch.tensor([1, 2, 3]).share(alice, bob, james, dtype="long")
    assert (
        x.child.dtype == "long"
        and x.child.field == 2 ** 64
        and isinstance(
            x.child.child["alice"].location.object_store.get_obj(
                x.child.child["alice"].id_at_location
            ),
            torch.LongTensor,
        )
        and (x.get() == torch.LongTensor([1, 2, 3])).all()
    )

    x = torch.tensor([4, 5, 6]).share(alice, bob, james, dtype="int")
    assert (
        x.child.dtype == "int"
        and x.child.field == 2 ** 32
        and isinstance(
            x.child.child["alice"].location.object_store.get_obj(
                x.child.child["alice"].id_at_location
            ),
            torch.IntTensor,
        )
        and (x.get() == torch.IntTensor([4, 5, 6])).all()
    )

    # With dtype custom
    x = torch.tensor([1, 2, 3]).share(alice, bob, james, dtype="custom", field=67)
    assert (
        x.child.dtype == "custom"
        and x.child.field == 67
        and isinstance(
            x.child.child["alice"].location.object_store.get_obj(
                x.child.child["alice"].id_at_location
            ),
            torch.IntTensor,
        )
        and (x.get() == torch.IntTensor([1, 2, 3])).all()
    )

    # With fix_prec
    x = torch.tensor([1.1, 2.2, 3.3]).fix_prec().share(alice, bob, james)
    assert (
        x.child.child.dtype == "long"
        and x.child.child.field == 2 ** 64
        and isinstance(
            x.child.child.child["alice"].location.object_store.get_obj(
                x.child.child.child["alice"].id_at_location
            ),
            torch.LongTensor,
        )
        and (x.get().float_prec() == torch.tensor([1.1, 2.2, 3.3])).all()
    )

    x = torch.tensor([4.1, 5.2, 6.3]).fix_prec(dtype="int").share(alice, bob, james)
    assert (
        x.child.child.dtype == "int"
        and x.child.child.field == 2 ** 32
        and isinstance(
            x.child.child.child["alice"].location.object_store.get_obj(
                x.child.child.child["alice"].id_at_location
            ),
            torch.IntTensor,
        )
        and (x.get().float_prec() == torch.tensor([4.1, 5.2, 6.3])).all()
    )


def test_garbage_collect_reconstruct(workers):
    bob, alice, james, me = (workers["bob"], workers["alice"], workers["james"], workers["me"])
    a = torch.ones(1, 5)
    a_sh = a.encrypt(workers=[alice, bob], crypto_provider=james)
    a_recon = a_sh.child.child.reconstruct()

    assert len(alice.object_store._objects) == 2
    assert len(bob.object_store._objects) == 2


def test_garbage_collect_move(workers):
    bob, alice, me = (workers["bob"], workers["alice"], workers["me"])
    a = torch.ones(1, 5).send(alice)
    b = a.copy().move(bob)

    assert len(alice.object_store._objects) == 1
    assert len(bob.object_store._objects) == 1


def test_garbage_collect_mul(workers):
    bob, alice, james, me = (workers["bob"], workers["alice"], workers["james"], workers["me"])
    a = torch.ones(1, 5)
    b = torch.ones(1, 5)

    a = a.encrypt(workers=[alice, bob], crypto_provider=james)
    b = b.encrypt(workers=[alice, bob], crypto_provider=james)

    for _ in range(3):
        c = a * b

    assert len(alice.object_store._objects) == 3
    assert len(bob.object_store._objects) == 3
