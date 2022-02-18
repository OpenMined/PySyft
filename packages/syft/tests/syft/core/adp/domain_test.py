# third party
import pytest

# syft absolute
import syft as sy
from syft.core.adp.scalar.phi_scalar import PhiScalar


# TODO @Tudor fix
@pytest.mark.xfail
def test_autodp_phiscalar_publish_domain(domain: sy.Domain) -> None:
    client = domain.get_root_client()

    x = PhiScalar(0, 0.01, 1).send(client, tags=["x"])
    y = PhiScalar(0, 0.02, 1).send(client, tags=["y"])
    z = PhiScalar(0, 0.02, 1).send(client, tags=["z"])

    p = x * x
    o = p + (y * y) + z
    z = o * o * o

    x_pub = z.publish(client, sigma=0.00001)

    print(x_pub, type(x_pub))
    print(x_pub.result)

    assert True is False  # to show output from errors
