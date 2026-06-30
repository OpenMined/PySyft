# syft-migration

Versioning and on-the-fly migration foundation for Syft objects.

Lets peers running different package versions exchange data by upgrading or downgrading
serialized objects to a version the other side understands.

## Building blocks

- `MigratableObject` — base for any versioned object (`canonical_name` + `version`),
  auto-registers via `__init_subclass__`.
- `PackageProtocolSchema` — the protocol surface of one release of one package
  (`protocol_name`, `package_name`, `package_version`, one version per object).
- `MigrationRegistry` — per-package registry of all object versions, migration edges, and
  the current + historical protocol schemas.
- `MigrationService` — upgrades/downgrades objects, including to the version a peer's
  package version supports.

## Dev

```bash
uv pip install -e packages/syft-migration
just test-unit-migration
```
