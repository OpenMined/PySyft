# stdlib
import time

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy

DOMAIN1_PORT = 9082


@pytest.mark.e2e
def test_get_timeout() -> None:
    domain_client = sy.old_login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    # send some data in
    x = np.array([1, 2, 3])
    x_ptr = x.send(domain_client)

    res = None
    try:
        res = x_ptr.get(timeout_secs=30)  # this should work but CI can be slow
    except Exception:
        pass

    # getting data with a low timeout works
    assert (res == x).all()

    # using the same pointer lets retry for 5 seconds
    start_time = time.time()
    TEST_TIMEOUT = 5
    res = None
    try:
        # reset the pointer and processing flag
        domain_client.processing_pointers[x_ptr.id_at_location] = True
        x_ptr._exhausted = False
        res = x_ptr.get(timeout_secs=TEST_TIMEOUT)
    except Exception:
        pass

    # after 5 seconds we stopped but didnt get anything
    assert time.time() > start_time + TEST_TIMEOUT
    assert res is None
