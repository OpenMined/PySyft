---
paths:
  - 'packages/syft-datasets/**'
  - 'packages/syft-rds/**'
---

# Dataset invariants

Facts that every change to datasets keeps true. A change that has to break one updates this file
in the same pull request and says why. `candidate` means not yet confirmed by the team. The
`syft-pr-feedback` skill checks pull requests against these.

- **DS-1 · A dataset is written in the current protocol by default.** An older layout is written
  only for a peer that reads it (`DatasetStorage.target_protocol_versions_for_peers()`).
  _(candidate · Koen, PR 9443)_
- **DS-2 · The dataset codec decides the disk layout.** `SyftDatasetManager` never branches on a
  protocol version. _(candidate · Pedro, PR 9443)_
- **DS-3 · syft shares collections and syft-rds shares datasets.** A collection is a set of files
  shared with many users; a dataset is one kind of collection. Dataset logic stays in
  syft-datasets and syft-rds. _(candidate · Koen, PR 9438)_
