# stdlib
from copy import copy

# third party
import pytest
import torch as th

# syft absolute
import syft as sy

DOMAIN_PORT = 9082


@pytest.mark.security
def test_store_object_mutation() -> None:
    """Check if the store object can be mutated by another user."""

    domain = sy.login(
        port=DOMAIN_PORT, email="info@openmined.org", password="changethis"
    )

    x = th.tensor([1, 2, 3])
    x_ptr = x.send(domain, pointable=True, tags=["visible"])

    y = th.tensor([3, 6, 9])
    y_ptr = y.send(domain, pointable=False, tags=["invisible"])
    y_ptr.id_at_location

    guest = sy.login(port=DOMAIN_PORT)
    guest_x = guest.store[x_ptr.id_at_location]

    guest_x.add_(guest_x)

    guest_y = copy(guest_x)

    guest_y.id_at_location = sy.common.UID.from_string(y_ptr.id_at_location.no_dash)

    guest_y.add_(guest_y)

    x_result = x_ptr.get(delete_obj=False)
    assert x_result == x
    y_result = y_ptr.get(delete_obj=False)
    assert y_result == y
