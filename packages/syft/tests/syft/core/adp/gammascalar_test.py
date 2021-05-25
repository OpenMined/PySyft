# syft absolute
from syft.core.adp.entity import Entity
from syft.core.adp.scalar import GammaScalar


def test_scalar() -> None:
    bob = GammaScalar(value=1, min_val=-2, max_val=2, entity=Entity(unique_name="Bob"))
    alice = GammaScalar(
        value=1, min_val=-1, max_val=1, entity=Entity(unique_name="Alice")
    )
    bob + alice
    bob - alice
    bob * alice
