"""A permission file keeps its meaning whatever case its name has.

macOS and Windows keep the case of a name but match it without case, so a
write to SYFT.PUB.YAML replaces the content of syft.pub.yaml. The event split
must classify that write as a permission event, or the DO sends it as ordinary
data and never recomputes who may read the files it governs.
"""

import uuid

from syft.sync.events.file_change_event import FileChangeEvent
from syft.sync.sync.datasite_owner_syncer import DatasiteOwnerSyncer


def _event(path: str) -> FileChangeEvent:
    return FileChangeEvent(
        id=uuid.uuid4(),
        path_in_datasite=path,
        datasite_email="owner@test.com",
        content="",
        submitted_timestamp=0.0,
        timestamp=0.0,
    )


def _split(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split by path. _split_events reads no instance state."""
    perm, data = DatasiteOwnerSyncer._split_events(None, [_event(p) for p in paths])
    return (
        [str(e.path_in_datasite) for e in perm],
        [str(e.path_in_datasite) for e in data],
    )


def test_permission_events_are_split_by_name_without_case():
    """A name that only ends with the constant stays a data event."""
    perm, data = _split(
        [
            "SYFT.PUB.YAML",
            "sub/Syft.Pub.Yaml",
            "syft.pub.yaml",
            "notsyft.pub.yaml",
            "sub/data.csv",
        ]
    )
    assert perm == ["SYFT.PUB.YAML", "sub/Syft.Pub.Yaml", "syft.pub.yaml"]
    assert data == ["notsyft.pub.yaml", "sub/data.csv"]
