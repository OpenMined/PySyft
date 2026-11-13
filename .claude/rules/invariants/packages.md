---
paths:
  - '**/pyproject.toml'
  - 'syft/version.py'
  - 'syft/__init__.py'
---

# Package invariants

Facts that every change to package metadata and dependencies keeps true. A change that has to break
one updates this file in the same pull request and says why. `candidate` means not yet confirmed by
the team. The `syft-pr-feedback` skill checks pull requests against these.

- **PKG-1 · Dependencies point one way.** syft-rds, syft-enclave and syft-bg build on syft; syft
  never imports them. _(candidate · Pedro, PR 9438)_
- **PKG-2 · Only the release job changes a version.** A pull request does not edit `version` in a
  `pyproject.toml` or `SYFT_VERSION` in `syft/version.py` (`docs/release.md`).
  _(candidate · Koen, PR 9443)_
