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

import syft  # noqa: F401 -- imports models and registers history
from syft.migrations import client_registry
from syft.sync.version.version_info import VersionInfo, VersionInfoV0

# Frozen on purpose. A change here means a change to the bootstrap file, so read
# the two rules above before editing this set.
V0_FIELDS = {
    "canonical_name",
    "version",
    "syft_client_version",
    "min_supported_syft_client_version",
    "protocol_version",
    "min_supported_protocol_version",
    "syft_client_install_source",
    "updated_at",
    "protocol_schemas",
    "extra",
}


def test_v0_fields_are_frozen():
    assert set(VersionInfoV0.model_fields) == V0_FIELDS, (
        "VersionInfoV0 changed. Every client reads this object first, so a "
        "removed or renamed field stops a client from parsing the version file "
        "of this one. Add a field with a default in a new version instead."
    )


def test_extra_is_opaque_to_syft():
    # syft carries whatever another package puts here and never interprets it,
    # so any JSON round-trips under any key.
    payload = {
        "attestation": {"kind": "made-up", "body": "anything"},
        "some-other-package": {"nested": [1, 2]},
    }
    info = VersionInfoV0(
        syft_client_version="0.10.2",
        min_supported_syft_client_version="0.10.2",
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
        extra=payload,
    )
    reloaded = VersionInfoV0.model_validate_json(info.model_dump_json())
    assert reloaded.extra == payload


def test_current_advertises_slim_schemas():
    schemas = VersionInfo.current().protocol_schemas
    # The client's own schema is always present; job/dataset only when the
    # optional packages are importable (both are in the workspace env, but
    # the code must degrade on client-only installs).
    assert "syft" in schemas
    assert set(schemas) <= {"syft", "syft-job", "syft-dataset"}
    client_schema = schemas["syft"]
    assert client_schema.version == client_registry.protocol_version
    assert client_schema.supported_versions == (
        client_registry.compute_protocol_schema().supported_versions
    )
    # Slim on the wire: no embedded per-object JSON schemas.
    for schema in schemas.values():
        assert schema.current_object_schemas == {}
