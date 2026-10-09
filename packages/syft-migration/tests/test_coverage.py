"""The coverage checks find an unregistered object and a missing migration.

``covpkg`` is a throwaway package next to these tests holding the shapes each
check has to catch. Every check returns findings, so an empty list is a pass.
"""

import covpkg
from covpkg import covered_registry, other_registry
from covpkg.objects import AbstractThing, FiledElsewhereV1, ThingV1, ThingV2

from syft_migration import (
    MigratableObject,
    MigrationRegistry,
    missing_downgrade_paths,
    missing_upgrade_paths,
    unregistered_objects,
    versioned_objects,
)


def _registry(protocol_version: str = "1") -> MigrationRegistry:
    return MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="1.0.0",
        protocol_version=protocol_version,
    )


# -- which classes count as versioned objects --------------------------------
def test_a_versioned_object_declared_in_the_package_init_is_scanned():
    # Its module is "covpkg", with no trailing dot, so a prefix-only scan misses it.
    names = {cls.__name__ for cls in versioned_objects(covpkg)}
    assert "DeclaredInInitV1" in names


def test_an_abstract_intermediate_is_not_a_versioned_object():
    # It leaves canonical_name/version required, so it is never registered and
    # asking for its identity would raise.
    assert AbstractThing not in versioned_objects(covpkg)


def test_the_scan_finds_the_objects_of_the_package_it_is_given():
    found = versioned_objects(covpkg)
    assert {ThingV1, ThingV2, FiledElsewhereV1} <= set(found)


# -- every versioned object is registered ------------------------------------
def test_an_object_registered_into_another_packages_registry_is_reported():
    # FiledElsewhereV1 lives in covpkg but named other_registry, which the base
    # class permits; only this check catches it.
    assert unregistered_objects(covered_registry, covpkg) == [FiledElsewhereV1]


def test_the_registry_that_does_hold_the_object_does_not_report_it():
    assert FiledElsewhereV1 not in unregistered_objects(other_registry, covpkg)


# -- every versioned object has migrations -----------------------------------
def test_a_registry_with_every_migration_reports_no_missing_paths():
    # covered_registry holds Thing 1<->2 both ways and a single-version object.
    assert missing_upgrade_paths(covered_registry) == []
    assert missing_downgrade_paths(covered_registry) == []


def test_a_missing_upgrade_to_latest_is_reported():
    registry = _registry()

    # Each class body registers itself; no migration is registered between them.
    class GapV1(MigratableObject, registry=registry):
        canonical_name: str = "Gap"
        version: str = "1"

    class GapV2(MigratableObject, registry=registry):
        canonical_name: str = "Gap"
        version: str = "2"

    assert missing_upgrade_paths(registry) == [("Gap", "1")]


def test_a_missing_downgrade_is_reported():
    registry = _registry()

    # Each class body registers itself.
    class StepV1(MigratableObject, registry=registry):
        canonical_name: str = "Step"
        version: str = "1"

    class StepV2(MigratableObject, registry=registry):
        canonical_name: str = "Step"
        version: str = "2"

    registry.register_migration(
        canonical_name="Step", from_version="1", to_version="2", fn=lambda obj: StepV2()
    )

    assert missing_upgrade_paths(registry) == []
    assert missing_downgrade_paths(registry) == [("Step", "2", "1")]


def test_a_downgrade_reached_through_an_intermediate_version_counts():
    registry = _registry()

    class ChainV1(MigratableObject, registry=registry):
        canonical_name: str = "Chain"
        version: str = "1"

    class ChainV2(MigratableObject, registry=registry):
        canonical_name: str = "Chain"
        version: str = "2"

    class ChainV3(MigratableObject, registry=registry):
        canonical_name: str = "Chain"
        version: str = "3"

    for lower, higher, made in ((1, 2, ChainV2), (2, 3, ChainV3)):
        registry.register_migration(
            canonical_name="Chain",
            from_version=str(lower),
            to_version=str(higher),
            fn=lambda obj, made=made: made(),
        )
    for higher, lower, made in ((3, 2, ChainV2), (2, 1, ChainV1)):
        registry.register_migration(
            canonical_name="Chain",
            from_version=str(higher),
            to_version=str(lower),
            fn=lambda obj, made=made: made(),
        )

    # 3 -> 1 is never registered; it is reached through version 2.
    assert missing_downgrade_paths(registry) == []


def test_version_ten_is_ordered_above_version_two():
    """A string sort puts "10" below "2" and would check the wrong direction."""
    registry = _registry()

    # Each class body registers itself.
    class BigV2(MigratableObject, registry=registry):
        canonical_name: str = "Big"
        version: str = "2"

    class BigV10(MigratableObject, registry=registry):
        canonical_name: str = "Big"
        version: str = "10"

    registry.register_migration(
        canonical_name="Big", from_version="2", to_version="10", fn=lambda obj: BigV10()
    )

    # Only the upgrade exists, so the missing downgrade is 10 -> 2. Ordering the
    # versions as strings would instead ask for 2 -> 10 and find it.
    assert missing_upgrade_paths(registry) == []
    assert missing_downgrade_paths(registry) == [("Big", "10", "2")]
