import syft as sy
import torch as th


def test_duet_send_and_get() -> None:
    duet = sy.Duet(host="127.0.0.1", port=5001)

    x = th.tensor([1, 2, 3])
    xp = x.send(duet)

    assert xp.id_at_location == x.id

    yp = xp + xp

    y = yp.get()
    assert ((x + x) == y).all()

    duet.stop()


def test_duet_searchable_functionality() -> None:
    duet = sy.Duet(host="127.0.0.1", port=5001)

    xp = th.tensor([1,2,3]).tag("some", "diabetes", "data").send(duet)
    xp2 = th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet, searchable=True)

    guest = sy.Duet(domain_url="http://127.0.0.1:5000/")

    assert len(guest.store) == 1
    assert len(duet.store) == 2

    assert guest.store[0].id_at_location == xp2.id_at_location

    duet.stop()
