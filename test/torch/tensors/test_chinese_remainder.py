import torch
import syft

# from syft.frameworks.torch.tensors.interpreters import syft.CRTTesor
# from syft import syft.CRTTesor


def test_wrap():
    """
    Test the .on() wrap functionality for syft.CRTTesor
    """

    res_3 = torch.tensor([[1, 2], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    res_7 = torch.tensor([[3, 4], [5, 6]]).fix_prec(field=7, precision_fractional=0)
    residues = {3: res_3, 7: res_7}

    x = syft.CRTTensor().on(residues, wrap=False).wrap()

    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, syft.CRTTensor)
    assert isinstance(x.child.child, dict)


def test__str__():
    t = torch.tensor([[3, 9], [4, 1]])
    crt = t.fix_precision(field=21, precision_fractional=0, storage="crt")

    assert isinstance(crt.__str__(), str)


def test_eq():
    t_a = torch.tensor([[3, 9], [4, 1]])
    t_b = torch.tensor([[3, 9], [4, 1]])
    t_c = torch.tensor([[2, 9], [4, 2]])

    crt_a = t_a.fix_precision(field=21, precision_fractional=0, storage="crt")
    crt_b = t_b.fix_precision(field=21, precision_fractional=0, storage="crt")
    crt_c = t_c.fix_precision(field=21, precision_fractional=0, storage="crt")

    eq_ab = (crt_a == crt_b).float_precision()
    eq_ac = (crt_a == crt_c).float_precision()

    assert (eq_ab == torch.tensor([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (eq_ac == torch.tensor([[0.0, 1.0], [1.0, 0.0]])).all()


def test_add():
    # TODO add tests for values > Q/2 and also wrapping
    t = torch.tensor([[1, 2], [3, 4]])

    crt_a = t.fix_precision(field=21, precision_fractional=0, storage="crt")
    crt_b = t.fix_precision(field=21, precision_fractional=0, storage="crt")

    result = crt_a + crt_b

    assert (result.float_precision() == torch.tensor([[2.0, 4.0], [6.0, 8.0]])).all()


def test_sub():
    t_a = torch.tensor([[4, 3], [2, 1]])
    t_b = torch.tensor([[1, 2], [3, 4]])

    crt_a = t_a.fix_precision(field=21, precision_fractional=0, storage="crt")
    crt_b = t_b.fix_precision(field=21, precision_fractional=0, storage="crt")

    result = crt_a - crt_b

    assert (result.float_precision() == torch.tensor([[3.0, 1.0], [-1.0, -3.0]])).all()


def test_mul():
    # 2 CRT tensors
    # TODO add tests for values > Q/2 and also wrapping
    t_a = torch.tensor([[1, -2], [3, -2]])
    t_b = torch.tensor([[1, 2], [3, -2]])

    crt_a = t_a.fix_precision(field=21, precision_fractional=0, storage="crt")
    crt_b = t_b.fix_precision(field=21, precision_fractional=0, storage="crt")

    result = crt_a * crt_b

    assert (result.float_precision() == torch.tensor([[1.0, -4.0], [9.0, 4.0]])).all()

    # With scalar
    result = 3 * crt_a

    assert (result.float_precision() == torch.tensor([[3.0, -6.0], [9.0, -6.0]])).all()


def test_send_and_get(workers):
    alice, bob = (workers["alice"], workers["bob"])

    t = torch.tensor([1, 2])
    crt = t.fix_precision(field=21, precision_fractional=0, storage="crt")

    to_alice = crt.send(alice)
    to_alice_id = to_alice.id_at_location

    assert to_alice_id in alice._objects

    to_bob_to_alice = to_alice.send(bob)
    to_bob_to_alice_id = to_bob_to_alice.id_at_location

    assert to_alice_id in alice._objects
    assert to_bob_to_alice_id in bob._objects

    to_alice_back = to_bob_to_alice.get()

    assert to_bob_to_alice_id not in bob._objects
    assert to_alice_id in alice._objects

    t_back = to_alice_back.get()

    assert to_alice_id not in alice._objects

    eq = (t_back == crt).float_precision()
    assert (eq == torch.tensor([1.0, 1.0])).all()


def test_share_and_get(workers):
    alice, bob, james = (workers["alice"], workers["bob"], workers["james"])

    t = torch.tensor([1, 2])
    crt = t.fix_precision(field=21, precision_fractional=0, storage="crt")
    copy = t.fix_precision(field=21, precision_fractional=0, storage="crt")

    shared = crt.share(alice, bob, crypto_provider=james)
    back = shared.get()

    assert (back.float_precision() == copy.float_precision()).all()
