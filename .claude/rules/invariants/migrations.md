---
paths:
  - '**/migrations/**'
  - '**/models/**'
  - '**/protocolcodecs/**'
  - 'scripts/*release*'
  - 'packages/*/scripts/*release*'
  - '.github/workflows/cd-*.yml'
---

# Migration invariants

Facts that every change to versioned objects, protocols and release artifacts keeps true. A change
that has to break one updates this file in the same pull request and says why. `candidate` means not
yet confirmed by the team. The `syft-pr-feedback` skill checks pull requests against these.

- **MIG-1 · A frozen release artifact is never edited.** The release job writes the files in
  `migrations/history/`, and a second export writes nothing (`docs/release.md`).
  _(candidate · docs/release.md)_
- **MIG-2 · A change to the schema of a versioned object is a new object version.** The drift check
  (`MigrationRegistry.find_schema_drift()`) fails otherwise
  (`packages/syft-migration/docs/object-versions.md`). _(candidate · docs)_
- **MIG-3 · A release fixture exists only for a released version.** Nothing under
  `migrations/p2p/fixtures/` names a version that is not on PyPI. _(candidate · Koen, PR 9443)_
