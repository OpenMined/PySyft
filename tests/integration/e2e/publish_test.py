# third party
import numpy as np
import pytest

# syft absolute
import syft as sy

DOMAIN1_PORT = 9082


@pytest.mark.e2e
def test_publish_with_bool_type_values(capfd) -> None:
    data_scientist = {
        "name": "Joker",
        "email": "joker@ace.com",
        "password": "iknowbatman",
        "budget": 10000,
    }

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    data_shape = (10000,)
    data1 = np.random.randint(1, 1000, data_shape)

    dataset1 = sy.Tensor(data1).annotate_with_dp_metadata(-1, 10000, data_subject="Jim")

    data1_ptr = dataset1.send(domain_client)

    domain_client.create_user(**data_scientist)

    ds_client = sy.login(
        email=data_scientist["email"],
        password=data_scientist["password"],
        port=DOMAIN1_PORT,
    )

    data1_ptr_ds = ds_client.store[data1_ptr.id_at_location]

    data_val_gt_100 = data1_ptr_ds > 100

    assert data_val_gt_100.public_dtype == bool
    assert data_val_gt_100.public_shape == data_shape

    mean_val = data1_ptr_ds.mean(keepdims=True)

    mean_gt_than_zero = mean_val > 0

    assert mean_gt_than_zero.public_dtype == bool
    assert mean_gt_than_zero.public_shape == (1,)

    # setting low sigma because True/False values are either 0 or 1.
    public_val = mean_gt_than_zero.publish(sigma=0.5)
    out, _ = capfd.readouterr()

    assert "WARNING" in out and "bool" in out
    TEST_TIMEOUT_SECS = 30  # increased timeout for arm64 tests in CI

    result = None

    try:
        # reset the pointer and processing flag
        result = public_val.get(timeout_secs=TEST_TIMEOUT_SECS)
    except Exception:
        pass

    assert public_val.exists
    assert result is not None

    # even though the published val is bool, the result is float cause
    # of noise addition. We don't convert back to bool intentionally.
    assert result.dtype == float
