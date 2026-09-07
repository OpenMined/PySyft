"""Object versions order by number, not as strings."""

import pytest

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    ProtocolSchema,
)


def _registry() -> MigrationRegistry:
    return MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="1.0.0",
        protocol_version="1",
    )


def _two_digit_registry() -> tuple[
    MigrationRegistry, type[MigratableObject], type[MigratableObject]
]:
    """A registry with version 2 and version 10 of the same object."""
    reg = _registry()

    class ThingV2(MigratableObject, registry=reg):
        canonical_name: str = "thing"
        version: str = "2"

    class ThingV10(MigratableObject, registry=reg):
        canonical_name: str = "thing"
        version: str = "10"
        extra: int = 0

    return reg, ThingV2, ThingV10


def test_latest_version_orders_by_number():
    reg, _, _ = _two_digit_registry()
    assert reg.latest_version(canonical_name="thing") == "10"


def test_computed_schema_freezes_the_highest_version():
    # find_schema_drift compares the frozen schema of the highest version. A
    # string order freezes version 2 and leaves version 10 unguarded.
    reg, _, thing_v10 = _two_digit_registry()
    schema = reg.compute_protocol_schema()
    assert schema.supported_versions == {"thing": ["2", "10"]}
    assert schema.current_object_schemas["thing"] == thing_v10.model_json_schema()


def test_current_schema_orders_by_number():
    schema = ProtocolSchema(
        protocol_name="p",
        version="1",
        supported_versions={"thing": ["2", "10"]},
    )
    assert schema.current_schema(canonical_name="thing") == "10"


@pytest.mark.parametrize("reverse", [False, True])
def test_from_objects_picks_the_highest_version(reverse):
    _, thing_v2, thing_v10 = _two_digit_registry()
    classes = [thing_v10, thing_v2] if reverse else [thing_v2, thing_v10]
    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="1",
        classes=classes,
    )
    assert schema.supported_versions == {"thing": ["2", "10"]}
    assert schema.current_object_schemas["thing"] == thing_v10.model_json_schema()


def test_upgradeable_path_targets_the_highest_version():
    reg, _, _ = _two_digit_registry()

    # Version 3 has no migration, so it cannot reach version 10. A string order
    # makes version 3 the latest and reports the path as trivially available.
    class ThingV3(MigratableObject, registry=reg):
        canonical_name: str = "thing"
        version: str = "3"

    reg.register_migration(
        canonical_name="thing",
        from_version="2",
        to_version="10",
        fn=lambda obj: obj,
    )
    assert reg.has_upgradeable_path_to_latest(canonical_name="thing", from_version="2")
    assert not reg.has_upgradeable_path_to_latest(
        canonical_name="thing", from_version="3"
    )


def test_non_numeric_object_version_is_rejected():
    reg = _registry()
    with pytest.raises(MigrationError):

        class ThingV1Patch(MigratableObject, registry=reg):
            canonical_name: str = "thing"
            version: str = "1.0"
