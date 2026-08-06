"""VersionInfo may only grow, because it is the bootstrap channel.

A peer reads SYFT_version.json before it knows anything else, so every supported
client must parse every newer file. Two rules follow, and neither is enforced by
the migration system:

- A field of an older version must not disappear or change name. An older reader
  requires it, and pydantic raises when it is absent.
- A field that a newer version adds must have a default. A newer reader must
  still parse a file that an older client wrote without that field.

Adding a field is safe on its own: pydantic ignores a field it does not know.
"""

import syft_client  # noqa: F401 -- imports models and registers history
from syft_client.sync.version.version_info import VersionInfoV1, VersionInfoV2

# Frozen on purpose. A change here means a change to the bootstrap file, so read
# the two rules above before editing this set.
V1_FIELDS = {
    "canonical_name",
    "version",
    "syft_client_version",
    "min_supported_syft_client_version",
    "protocol_version",
    "min_supported_protocol_version",
    "syft_client_install_source",
    "updated_at",
    "attestation_token",
}

V2_ADDS = {"protocol_schemas"}


def test_v1_fields_are_frozen():
    assert set(VersionInfoV1.model_fields) == V1_FIELDS, (
        "VersionInfoV1 changed. A client that speaks protocol 0 reads this "
        "object, so a removed or renamed field stops that client from parsing "
        "the version file of this one."
    )


def test_v2_keeps_every_v1_field():
    missing = V1_FIELDS - set(VersionInfoV2.model_fields)
    assert not missing, (
        f"VersionInfoV2 dropped {sorted(missing)}. A reader of V1 requires these "
        "fields, so V2 must keep them."
    )


def test_v2_adds_only_the_expected_fields():
    assert set(VersionInfoV2.model_fields) - V1_FIELDS == V2_ADDS


def test_fields_added_after_v1_have_a_default():
    # A file written by an older client carries none of these, so a reader of the
    # newer version must supply a value.
    for name in set(VersionInfoV2.model_fields) - V1_FIELDS:
        assert not VersionInfoV2.model_fields[name].is_required(), (
            f"VersionInfoV2.{name} is required. A version file written before "
            "this field existed would then fail to parse."
        )


def test_a_file_without_the_v2_fields_still_parses():
    written_by_an_older_client = VersionInfoV1(
        syft_client_version="0.1.117",
        min_supported_syft_client_version="0.1.93",
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
    ).model_dump(exclude={"canonical_name", "version"})

    loaded = VersionInfoV2.model_validate(written_by_an_older_client)
    assert loaded.protocol_schemas == {}
