"""Every versioned syft object is known to the client registry."""

import syft
from syft.migrations.registry import client_registry
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
    FileChangeEventsMessageV1,
)
from syft.sync.messages.proposed_filechange import (
    ProposedFileChangesMessage,
    ProposedFileChangesMessageV1,
)
from syft.sync.version.version_info import VersionInfo, VersionInfoV2
from syft_migration import unregistered_objects, versioned_objects

OBJECTS = {"VersionInfo", "FileChangeEventsMessage", "ProposedFileChangesMessage"}


def test_versioned_objects_registered_and_aliased():
    # Every object has at least one version registered in the client registry.
    for canonical_name in OBJECTS:
        assert client_registry.versions(canonical_name)

    # The current-version aliases resolve to the latest class of each object.
    # VersionInfo is the one that has moved past V1.
    assert VersionInfo is VersionInfoV2
    assert FileChangeEventsMessage is FileChangeEventsMessageV1
    assert ProposedFileChangesMessage is ProposedFileChangesMessageV1

    # The protocol schema covers every object and resolves a current version.
    schema = client_registry.compute_protocol_schema()
    assert OBJECTS <= set(schema.supported_versions)
    for canonical_name in OBJECTS:
        assert schema.current_schema(canonical_name=canonical_name)


def test_all_migratable_objects_in_package_are_registered():
    # The scan imports every syft module, so it sees objects that nothing else
    # imports. It also catches an object filed into syft-job's or
    # syft-dataset's registry instead of this one.
    assert len(versioned_objects(syft)) >= len(OBJECTS)
    assert unregistered_objects(client_registry, syft) == []
