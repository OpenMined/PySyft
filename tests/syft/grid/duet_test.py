# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException


@pytest.mark.slow
def test_duet_send_and_get() -> None:
    duet = sy.Duet(host="127.0.0.1", port=5001)

    x = th.tensor([1, 2, 3])
    xp = x.send(duet)

    assert xp.id_at_location == x.id

    yp = xp + xp

    y = yp.get()
    assert ((x + x) == y).all()

    duet.stop()


@pytest.mark.slow
def test_duet_searchable_functionality() -> None:
    duet = sy.Duet(host="127.0.0.1", port=5001)

    xp = th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet)
    xp2 = (
        th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet, searchable=True)
    )

    guest = sy.Duet(domain_url="http://127.0.0.1:5001/")

    assert len(guest.store) == 1
    assert len(duet.store) == 2

    assert guest.store[0].id_at_location == xp2.id_at_location

    del xp
    del xp2

    duet.stop()


@pytest.mark.slow
def test_duet_exception_catching() -> None:
    try:
        duet = sy.Duet(host="127.0.0.1", port=5001)

        xp = th.tensor([1, 2, 3]).tag("some", "diabetes", "data").send(duet)
        xp2 = (
            th.tensor([1, 2, 3])
            .tag("some", "diabetes", "data")
            .send(duet, searchable=True)
        )

        guest = sy.Duet(domain_url="http://127.0.0.1:5001/")

        assert len(guest.store) == 1
        assert len(duet.store) == 2

        guest.store[0].get()  # raises AuthorizationException

        assert True is False  # we should never get here

    except Exception as e:
        print("Failed", type(e), e)
        assert isinstance(e, AuthorizationException)

    finally:
        del xp
        del xp2

        duet.stop()
