# stdlib
from copy import copy
import time

# third party
import pytest
import torch as th

# syft absolute
import syft as sy

DOMAIN1_PORT = 9082


@pytest.mark.security
def test_store_object_mutation() -> None:
    """Check if the store object can be mutated by another user."""

    domain = sy.login(
        port=DOMAIN1_PORT, email="info@openmined.org", password="changethis"
    )

    x = th.tensor([1, 2, 3])
    x_ptr = x.send(domain, pointable=True, tags=["visible"])

    y = th.tensor([3, 6, 9])
    y_ptr = y.send(domain, pointable=False, tags=["invisible"])
    y_ptr.id_at_location

    guest = sy.login(port=DOMAIN1_PORT)
    guest_x = guest.store[x_ptr.id_at_location]

    guest_x.add_(guest_x)

    guest_y = copy(guest_x)

    guest_y.id_at_location = sy.common.UID.from_string(y_ptr.id_at_location.no_dash)

    guest_y.add_(guest_y)

    # Guest user should not be able to mutate objects that don't belong to them
    x_result = x_ptr.get(delete_obj=False)
    assert all(x_result == x) is True

    y_result = y_ptr.get(delete_obj=False)
    assert all(y_result == y) is True

    # Domain owner should be able to mutate their own objects
    x_ptr.add_(x_ptr)

    # A sleep is needed because the `RunClassMethodAction` is async, and therefore if
    # do `.get`, then the value returned by the `x_ptr` could be stale while `RunClassMethodAction`
    # is still being executed.
    time.sleep(1)
    new_result = x_ptr.get(delete_obj=False)
    assert all(new_result == (x + x)) is True
