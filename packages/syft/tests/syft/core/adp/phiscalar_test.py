# third party
import pytest
from pytest import approx

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.adp.scalar import GammaScalar
from syft.core.adp.scalar import PhiScalar


def test_phiscalar() -> None:
    x = PhiScalar(0, 0.01, 1)

    assert x.min_val == 0
    assert x.value == 0.01
    assert x.max_val == 1
    assert x.ssid == str(x.poly)

    ent = Entity(name="test")
    y = PhiScalar(0, 0.01, 1, entity=ent)

    assert y.min_val == 0
    assert y.value == 0.01
    assert y.max_val == 1
    assert y.ssid == str(y.poly)
    assert y.entity == ent


@pytest.mark.xfail
def test_phiscalar_pointer(client: sy.VirtualMachineClient) -> None:
    x = PhiScalar(0, 0.01, 1)
    y = x + x
    y_gamma = y.gamma

    assert isinstance(y_gamma, GammaScalar)
    assert y_gamma.min_val == 0.0
    assert y_gamma.value == 0.02
    assert y_gamma.max_val == 2.0

    x_ptr = x.send(client, tags=["x"])
    y_ptr = x_ptr + x_ptr

    # can't retrieve the phiscalar because the poly is not serde yet and the lookup
    # to resolve the poly is in the remote systems memory so lets grab gamma
    y_gamma_ptr = y_ptr.gamma
    y_gamma2 = y_gamma_ptr.get()

    assert y_gamma2.min_val == y_gamma.min_val  # TODO Fix this underflow
    assert y_gamma2.value + y_gamma.value == approx(2 * y_gamma.value)
    assert y_gamma2.max_val == y_gamma.max_val
