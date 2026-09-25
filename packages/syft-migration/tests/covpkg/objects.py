from syft_migration import MigratableObject

from . import covered_registry, other_registry


class ThingV1(MigratableObject, registry=covered_registry):
    canonical_name: str = "Thing"
    version: str = "1"
    name: str = ""


class ThingV2(MigratableObject, registry=covered_registry):
    canonical_name: str = "Thing"
    version: str = "2"
    name: str = ""
    owner: str = ""


class AbstractThing(MigratableObject):
    """Leaves the identity fields required, so it is not a versioned object."""

    name: str = ""


class FiledElsewhereV1(MigratableObject, registry=other_registry):
    canonical_name: str = "FiledElsewhere"
    version: str = "1"


covered_registry.register_migration(
    canonical_name="Thing",
    from_version="1",
    to_version="2",
    fn=lambda obj: ThingV2(name=obj.name),
)
covered_registry.register_migration(
    canonical_name="Thing",
    from_version="2",
    to_version="1",
    fn=lambda obj: ThingV1(name=obj.name),
)
