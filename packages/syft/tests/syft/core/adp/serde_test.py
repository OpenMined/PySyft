# third party
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.adp.scalar import BaseScalar
from syft.core.adp.scalar import GammaScalar
from syft.core.adp.scalar import IntermediatePhiScalar
from syft.core.adp.scalar import IntermediateScalar
from syft.core.adp.scalar import PhiScalar

scalar_object_tests = [
    IntermediateScalar(poly=None),
    IntermediatePhiScalar(poly=None, entity=Entity(name="test")),
    BaseScalar(min_val=1, value=2, max_val=3, entity=Entity(name="test")),
    BaseScalar(min_val=None, value=None, max_val=None, entity=Entity(name="test")),
    BaseScalar(min_val=1, value=None, max_val=3, entity=Entity(name="test")),
    PhiScalar(min_val=0, value=1, max_val=2, entity=Entity(name="test")),
    GammaScalar(min_val=0, value=1, max_val=2, entity=Entity(name="test")),
]


@pytest.mark.parametrize("scalar", scalar_object_tests)
def test_serde_scalar(scalar: BaseScalar) -> None:
    protobuf_obj = serialize(scalar)

    deserialized = deserialize(protobuf_obj, from_proto=True)

    for field in [
        "id",
        "_gamma",
        "name",
        "max_value",
        "min_value",
        "value",
    ]:  # add poly support it
        if hasattr(scalar, field):
            assert getattr(scalar, field) == getattr(deserialized, field)
            assert isinstance(
                getattr(scalar, field), type(getattr(deserialized, field))
            )

    if hasattr(scalar, "entity"):
        assert scalar.entity.name == deserialized.entity.name
        assert scalar.entity.id == deserialized.entity.id
