# third party
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.adp.scalar.abstract.base_scalar import BaseScalar
from syft.core.adp.scalar.abstract.intermediate_scalar import IntermediateScalar

# from syft.core.adp.scalar.gamma_scalar import GammaScalar
from syft.core.adp.scalar.intermediate_phi_scalar import IntermediatePhiScalar
from syft.core.adp.scalar.phi_scalar import PhiScalar

scalar_object_tests = [
    IntermediateScalar(poly=None),
    IntermediatePhiScalar(poly=None, entity=Entity(name="test")),
    BaseScalar(min_val=1, value=2, max_val=3, entity=Entity(name="test")),
    # BaseScalar(min_val=None, value=None, max_val=None, entity=Entity(name="test")),
    BaseScalar(min_val=1, value=None, max_val=3, entity=Entity(name="test")),
    PhiScalar(min_val=0, value=1, max_val=2, entity=Entity(name="test")),
    # GammaScalar(min_val=0, value=1, max_val=2, entity=Entity(name="test"), prime=3),
]


@pytest.mark.parametrize("scalar", scalar_object_tests)
def test_serde_scalar(scalar: BaseScalar) -> None:
    protobuf_obj = serialize(scalar, to_proto=True)
    deserialized = deserialize(protobuf_obj, from_proto=True)

    for field in [
        "id",
        "max_val",
        "value",
        "min_val",
        "_gamma",
        "name",
        "entity",
        "ssid",
        "poly",
    ]:  # add poly support it
        if hasattr(scalar, field):
            assert getattr(scalar, field) == getattr(deserialized, field)
            assert isinstance(
                getattr(scalar, field), type(getattr(deserialized, field))
            )

    if hasattr(scalar, "entity"):
        assert scalar.entity.name == deserialized.entity.name
