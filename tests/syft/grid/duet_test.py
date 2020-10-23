# stdlib
from typing import Tuple

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException


@pytest.mark.slow
def test_duet_send_and_get(duet_wrapper: Tuple[sy.Duet, str]) -> None:
    duet, _ = duet_wrapper
    x = th.tensor([1, 2, 3])
    xp = x.send(duet)

    assert xp.id_at_location == x.id

    yp = xp + xp

    y = yp.get()
    assert ((x + x) == y).all()


@pytest.mark.slow
def test_duet_searchable_functionality(duet_wrapper: Tuple[sy.Duet, str]) -> None:
    duet, url = duet_wrapper
    xp = th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet)
    xp2 = (
        th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet, searchable=True)
    )

    guest = sy.Duet(domain_url=url)

    assert len(guest.store) == 1
    assert len(duet.store) == 2

    assert guest.store[0].id_at_location == xp2.id_at_location

    del xp
    del xp2


@pytest.mark.slow
def test_duet_exception_catching(duet_wrapper: Tuple[sy.Duet, str]) -> None:
    duet, url = duet_wrapper
    xp = th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet)
    xp2 = (
        th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet, searchable=True)
    )

    guest = sy.Duet(domain_url=url)

    assert len(guest.store) == 1
    assert len(duet.store) == 2

    with pytest.raises(AuthorizationException):
        guest.store[0].get()

    del xp
    del xp2
