# syft absolute
from syft.lib.adp.entity import Entity
from syft.lib.adp.scalar import Scalar


def test_scalar() -> None:
    bob = Scalar(value=1, min_val=-2, max_val=2, ent=Entity(name="Bob"))
    alice = Scalar(value=1, min_val=-1, max_val=1, ent=Entity(name="Alice"))
    bob + alice
