# syft absolute
import syft as sy
from syft.lib.adp.scalar import PhiScalar


def test_autodp_phiscalar_publish_domain() -> None:
    bob_domain = sy.Domain(name="Bob's Domain")
    client = bob_domain.get_root_client()

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
