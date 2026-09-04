"""Generate a p2p backward-compatibility fixture for the current syft release.

Run on EVERY release, at the released commit:

    git checkout v<released version>
    uv run python scripts/generate_release_fixture.py

The fixture name comes from SYFT_VERSION in the tree. The release job
publishes the version on the branch, tags it, then bumps. A run after the bump
therefore names the fixture after the next version, which is not published yet.

Writes the serialized artifacts exactly as this release produces them, into

    tests/migrations/p2p/fixtures/syft-<version>-protocol<p>/
        SYFT_version.json                       # the published version file
        msgv2_<...>.tar.gz                      # a proposed-changes message (DS -> DO)
        syfteventsmessagev3_<...>.tar.gz        # an events message (DO -> watchers)

Unlike syft-job there is no local SyftBox tree to snapshot (storage is the
Google Drive transport), so fixtures are directories of captured blobs; future
releases loop over them (test_older_protocol_compatibility.py) to prove they
can still read and round-trip older serialized data.

Protocol 0 / release 0.1.117 predates this script; its fixture
(syft_client-0.1.117-protocol0) is hand-authored, like protocol-0.json.
"""

import sys
from pathlib import Path

from syft.migrations.registry import SYFT_CLIENT_PROTOCOL_VERSION
from syft.sync.events.file_change_event import (
    FileChangeEvent,
    FileChangeEventsMessage,
)
from syft.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
)
from syft.sync.version.version_info import VersionInfo
from syft.version import SYFT_VERSION

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"

FIXTURES_DIR = (
    Path(__file__).resolve().parents[1] / "tests" / "migrations" / "p2p" / "fixtures"
)


def build_version_info() -> VersionInfo:
    # Not VersionInfo.current(): the detected install source is an absolute
    # local path on dev machines, which must not leak into a committed fixture.
    return VersionInfo.current().model_copy(
        update={"syft_client_install_source": "pip"}
    )


def build_proposed_message() -> ProposedFileChangesMessage:
    return ProposedFileChangesMessage(
        sender_email=DS_EMAIL,
        proposed_file_changes=[
            ProposedFileChange(
                path_in_datasite="data/notes.txt",
                content="hello from the release fixture",
                datasite_email=DO_EMAIL,
            ),
            ProposedFileChange(
                path_in_datasite="data/blob.bin",
                content=b"\x00\x01\x02fixture-binary",
                datasite_email=DO_EMAIL,
            ),
            ProposedFileChange(
                path_in_datasite="data/removed.txt",
                content=None,
                old_hash="0" * 64,
                is_deleted=True,
                datasite_email=DO_EMAIL,
            ),
        ],
    )


def build_events_message(
    proposed: ProposedFileChangesMessage,
) -> FileChangeEventsMessage:
    events = [
        FileChangeEvent.from_proposed_filechange(change)
        for change in proposed.proposed_file_changes
    ]
    return FileChangeEventsMessage(events=events)


def main() -> None:
    # Any fixture for this version (any protocol) means the version was
    # already released; a released version's serialized form is frozen.
    existing = sorted(FIXTURES_DIR.glob(f"syft-{SYFT_VERSION}-protocol*"))
    if existing:
        sys.exit(
            f"{existing[0]} already exists — fixtures are frozen once written. "
            "Bump SYFT_VERSION before generating."
        )
    target = FIXTURES_DIR / (
        f"syft-{SYFT_VERSION}-protocol{SYFT_CLIENT_PROTOCOL_VERSION}"
    )
    target.mkdir(parents=True)

    (target / "SYFT_version.json").write_text(build_version_info().to_json())

    proposed = build_proposed_message()
    (target / proposed.message_filename.as_string()).write_bytes(
        proposed.as_compressed_data()
    )

    events = build_events_message(proposed)
    (target / events.message_filepath.as_string()).write_bytes(
        events.as_compressed_data()
    )

    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
