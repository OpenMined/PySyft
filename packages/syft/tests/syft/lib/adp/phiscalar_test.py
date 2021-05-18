# syft absolute
from syft.lib.adp.entity import Entity
from syft.lib.adp.scalar import PhiScalar


def test_phiscalar() -> None:
    x = PhiScalar(0, 0.01, 1)

    assert x.min_val == 0
    assert x.value == 0.01
    assert x.max_val == 1
    assert x.ssid == str(x.poly)

    ent = Entity(unique_name="test")
    y = PhiScalar(0, 0.01, 1, entity=ent)

    assert y.min_val == 0
    assert y.value == 0.01
    assert y.max_val == 1
    assert y.ssid == str(y.poly)
    assert y.entity == ent
