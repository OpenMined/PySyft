# Add a Version to a Versioned Object

## Overview

A versioned object is a pydantic model. A peer reads it from disk, or receives it over the network. Each version of the object is a separate class.

`syft-migration` holds every version and every migration between the versions. Therefore the new release reads a file or a message that a different release wrote.

Every path in this document is relative to the root of the repository.

Three packages hold versioned objects. Each package has its own registry.

| Package        | Registry           | Import from                |
| -------------- | ------------------ | -------------------------- |
| `syft`         | `client_registry`  | `syft.migrations.registry` |
| `syft-job`     | `job_registry`     | `syft_job.migrations`      |
| `syft-dataset` | `dataset_registry` | `syft_datasets.migrations` |

An object version is an incrementing integer held as a string: `"1"`, `"2"`, `"3"`. It is not a semver. It has no relation to the package version.

## When to add a new version

A release freezes an object version forever. Each release artifact stores the JSON schema of the object versions of that release. `find_schema_drift()` compares every live class against every frozen schema. A change to a released class is a defect. Add a new version instead.

These changes need a new version:

- add a field, remove a field, or rename a field
- change the type of a field
- change the default value of a field
- change the `description` or the `title` of a field
- change the docstring of the class

These changes do not need a new version:

- add or change a method, a classmethod, or a property
- change a comment

> [!WARNING]
>
> The docstring rule is easy to miss. Pydantic copies the docstring of the class
> into `model_json_schema()` as `description`. It copies the `description` and
> the `title` of each field there too. Therefore all of that prose is part of
> the frozen schema. `protocol-0.json` holds the frozen docstrings of
> `VersionInfoV1`, `ProposedFileChangesMessageV1`, and
> `FileChangeEventsMessageV1`. A correction to one of those docstrings fails
> `tests/migrations/unit/test_history_artifacts.py`. Put new prose in a comment,
> or in the new version.

## Where the files go

`syft-job` and `syft-dataset` keep one file for each version. The package `__init__.py` holds the current-version alias.

```
models/dataset/
├── __init__.py     # Dataset = DatasetV1
├── v1.py           # class DatasetV1
└── v2.py           # class DatasetV2, and the migrations between v1 and v2
```

The `syft` package keeps `VersionInfo` in one file, `syft/sync/version/version_info.py`. That file holds every version, both migrations, and the alias. Either layout is correct. Keep both migrations next to the new class, because the two edges change together.

## Steps

1. **Add the class.** Subclass the previous version, then pin the new `version` as a field default. The new class inherits the registry of the previous version. Do not pass `registry=` again.
2. **Register a migration in both directions.** Use `@registry.migration(canonical_name, from_version, to_version)`.
3. **Set the current-version alias to the new class.** Callers then always hold the latest version.
4. **Add a test fixture for the new version.** `syft-job` and `syft-dataset` need `packages/<package>/tests/migrations/unit/fixtures/<CanonicalName>/v<n>.yaml`.
5. **Check the protocol version constant.** The section below gives the rule.
6. **Run the test suite of the package that holds the object.** Then run the other three suites.

## Worked example

`VersionInfo` is the only object in the repository with two versions. V2 adds a field. Therefore the upgrade migration sets a default value for the field, and the downgrade migration removes it.

`VersionInfo` is also a special case. A peer reads `SYFT_version.json` to learn the protocol that this client speaks. Every supported client must parse every newer version of that file. A new version of `VersionInfo` can therefore add an optional field only.

Other objects have no such limit. `DatasetV2` can add a required field, because the upgrade migration builds the object and gives the field a value.

```python
class VersionInfoV2(VersionInfoV1):
    """V2 adds the protocol schemas this client speaks (client, job, dataset)."""

    version: str = "2"

    protocol_schemas: dict[str, ProtocolSchema] = Field(default_factory=dict)


@client_registry.migration("VersionInfo", "1", "2")
def _version_info_v1_to_v2(obj: VersionInfoV1) -> VersionInfoV2:
    # A v1 file says nothing about package protocols: empty schemas, meaning
    # "unknown speaker" to consumers.
    return VersionInfoV2.model_validate(
        obj.model_dump(exclude={"canonical_name", "version"})
    )


@client_registry.migration("VersionInfo", "2", "1")
def _version_info_v2_to_v1(obj: VersionInfoV2) -> VersionInfoV1:
    return VersionInfoV1.model_validate(
        obj.model_dump(exclude={"canonical_name", "version", "protocol_schemas"})
    )


# Current-version alias: callers always work with the latest VersionInfo.
VersionInfo = VersionInfoV2
```

Both migrations exclude `canonical_name` and `version` from the dump. Each class pins its own identity as a field default. If a migration passes the old pair, the new object gets the wrong version. The downgrade migration also excludes the field that V1 does not have.

## Register both directions

`migration_path()` is a breadth-first search over the registered edges. It never infers an inverse. If a downgrade edge is absent, this release cannot serve a peer that reads the lower version.

A migration for every pair of versions is not necessary. A path through an intermediate version is enough. Version 3 migrates to version 1 through version 2, with no 3-to-1 edge.

Order versions with the integer key of the registry. Do not use a string sort, because a string sort puts `"10"` before `"2"`.

## The protocol version constant

The protocol version names the layout on disk and on the network. A new object version changes `supported_versions` in the registry. Therefore the protocol version constant must be above the newest released protocol.

- The current protocol is not yet released. The constant is already above the newest released protocol, so a new object version needs no bump.
- The current protocol N is released. The next object version needs protocol N+1, so bump the constant.

| Package        | Constant                       | File                                                              |
| -------------- | ------------------------------ | ----------------------------------------------------------------- |
| `syft`         | `SYFT_CLIENT_PROTOCOL_VERSION` | `syft/migrations/registry.py`                                     |
| `syft-job`     | `JOB_PROTOCOL_VERSION`         | `packages/syft-job/src/syft_job/migrations/registry.py`           |
| `syft-dataset` | `DATASET_PROTOCOL_VERSION`     | `packages/syft-datasets/src/syft_datasets/migrations/registry.py` |

Two checks find a mistake:

- `protocol_bump_missing()` — the protocol changed after the newest released protocol, but the constant is the same.
- `protocol_changed_without_bump()` — the protocol changed against the released artifact for the current constant.

Each package runs both checks in its own export script, and the script then stops the release:

- `scripts/export_release_artifact.py` for `syft`
- `packages/syft-job/scripts/export_release_artifact.py` for `syft-job`
- `packages/syft-datasets/scripts/export_release_artifact.py` for `syft-dataset`

Both checks compare object versions only. A change to the layout that adds no object version is invisible to both checks. Therefore bump the constant by hand for a path change or a folder rename.

Do not raise `min_supported_protocol_version`. It is the oldest protocol that the package still reads. A higher value removes support for every peer below it.

## What the tests check

Each package runs the same checks against its own registry.

- `test_objects_registered.py` — every versioned object of the package is in the registry of that package. This check finds a class that is in the registry of a different package.
- `test_upgrade_paths.py` — every registered version reaches the latest version and every lower version. A migration edge that is absent fails here, and not in production.
- `test_history_artifacts.py` — no released object version drifted from its frozen schema.

`syft-job` and `syft-dataset` also run `test_migrations.py`. It loads the fixture of every registered version and upgrades it to the latest version. It also downgrades the latest version to every registered version. A new version with no fixture fails this test.

```bash
just test-unit-migration      # syft-migration
just test-client-migrations   # syft (tests/migrations)
just test-unit-job            # syft-job
just test-unit-datasets       # syft-dataset
```

## Checklist

- [ ] The new class subclasses the previous version and pins the new `version`.
- [ ] A migration exists in both directions.
- [ ] The upgrade migration sets every new field, and the downgrade migration removes it.
- [ ] The current-version alias refers to the new class.
- [ ] A fixture exists for the new version, for `syft-job` and `syft-dataset`.
- [ ] The protocol version constant is above the newest released protocol.
- [ ] No released class changed. A class docstring and a field `description` are part of the schema.
- [ ] All four test suites pass.
