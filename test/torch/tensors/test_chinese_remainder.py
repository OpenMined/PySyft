import torch
import syft


def test__str__():
    t = torch.tensor([[3, 9], [4, 1]])
    crt = t.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

    assert isinstance(crt.__str__(), str)


def test_eq():
    t_a = torch.tensor([[3, 9], [4, 1]])
    t_b = torch.tensor([[3, 9], [4, 1]])
    t_c = torch.tensor([[2, 9], [4, 2]])

    crt_a = t_a.fix_precision(field_type="int100", precision_fractional=2, storage="crt")
    crt_b = t_b.fix_precision(field_type="int100", precision_fractional=2, storage="crt")
    crt_c = t_c.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

    eq_ab = (crt_a == crt_b).float_precision()
    eq_ac = (crt_a == crt_c).float_precision()

    assert (eq_ab == torch.tensor([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (eq_ac == torch.tensor([[0.0, 1.0], [1.0, 0.0]])).all()


def test__neg__():
    t = torch.tensor([[1, -3], [3, -2]])
    crt = t.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

    neg = -crt

    assert (neg.float_precision() == torch.tensor([[-1.0, 3.0], [-3.0, 2.0]])).all()


def test_add():
    t_a = torch.tensor([[1, 2], [3, -4]])
    t_b = torch.tensor([[1, -3], [3, -2]])

    crt_a = t_a.fix_precision(field_type="int100", precision_fractional=2, storage="crt")
    crt_b = t_b.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

    result = crt_a + crt_b

    assert (result.float_precision() == torch.tensor([[2.0, -1.0], [6.0, -6.0]])).all()

    # With scalars
    result = crt_a + 1

    assert (result.float_precision() == torch.tensor([[2.0, 3.0], [4.0, -3.0]])).all()


def test_sub():
    t_a = torch.tensor([[1, 2], [5, -4]])
    t_b = torch.tensor([[1, -3], [3, -2]])

    crt_a = t_a.fix_precision(field_type="int100", precision_fractional=2, storage="crt")
    crt_b = t_b.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

    result = crt_a - crt_b

    assert (result.float_precision() == torch.tensor([[0.0, 5.0], [2.0, -2.0]])).all()

    # With scalars
    result_a = crt_a - 1
    result_b = 1 - crt_a

    assert (result_a.float_precision() == torch.tensor([[0.0, 1.0], [4.0, -5.0]])).all()
    assert (result_b.float_precision() == torch.tensor([[0.0, -1.0], [-4.0, 5.0]])).all()


def test_mul():
    t_a = torch.tensor([[1, -2], [3, -2]])
    t_b = torch.tensor([[1, 2], [3, -2]])

    crt_a = t_a.fix_precision(field_type="int100", precision_fractional=2, storage="crt")
    crt_b = t_b.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

    result = crt_a * crt_b

    assert (result.float_precision() == torch.tensor([[1.0, -4.0], [9.0, 4.0]])).all()

    # With scalar
    result = 3 * crt_a

    assert (result.float_precision() == torch.tensor([[3.0, -6.0], [9.0, -6.0]])).all()


def test_send_and_get(workers):
    alice, bob = (workers["alice"], workers["bob"])

    t = torch.tensor([1, 2])
    crt = t.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

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
    crt = t.fix_precision(field_type="int100", precision_fractional=2, storage="crt")
    copy = t.fix_precision(field_type="int100", precision_fractional=2, storage="crt")

    shared = crt.share(alice, bob, crypto_provider=james)
    back = shared.get()

    assert (back.float_precision() == copy.float_precision()).all()
