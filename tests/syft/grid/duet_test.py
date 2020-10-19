# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException


@pytest.mark.slow
def test_duet_send_and_get(duet: sy.Duet) -> None:
    x = th.tensor([1, 2, 3])
    xp = x.send(duet)

    assert xp.id_at_location == x.id

    yp = xp + xp

    y = yp.get()
    assert ((x + x) == y).all()


@pytest.mark.slow
def test_duet_searchable_functionality(duet: sy.Duet) -> None:
    xp = th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet)
    xp2 = (
        th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet, searchable=True)
    )

    guest = sy.Duet(domain_url="http://127.0.0.1:5002/")

    assert len(guest.store) == 1
    assert len(duet.store) == 2

    assert guest.store[0].id_at_location == xp2.id_at_location

    del xp
    del xp2


@pytest.mark.slow
def test_duet_exception_catching(duet: sy.Duet) -> None:
    xp = th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet)
    xp2 = (
        th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet, searchable=True)
    )

    guest = sy.Duet(domain_url="http://127.0.0.1:5001/")

    assert len(guest.store) == 1
    assert len(duet.store) == 2

    try:
        guest.store[0].get()  # raises AuthorizationException

        assert False  # we should never get here

    except AuthorizationException:
        pass
    finally:
        del xp
        del xp2
