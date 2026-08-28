# Release Process

## Overview

Releases are managed through dedicated release branches. The mono repo release job handles bumping versions and pushing tags for all individual packages automatically.

## Steps

1. **Create a release branch** from `dev` (the default branch; `main` is the frozen legacy-PySyft branch), dont include the patch version in the semver, so we can hotfix patches on the same branch (e.g. `release/v0.10`). If you are patching, re-use the branch:
   ```bash
   git fetch origin && git checkout -b release/vX.Y origin/dev && git push -u origin release/vX.Y   # patches: git checkout release/vX.Y
   ```
2. **Run the release workflow.** You can trigger frmo github UI from the Actions tab. In most cases, release the mono repo — this releases all individual packages (`syft-permissions`, `syft-perms`, `syft-migration`, `syft-dataset`, `syft-job`, `syft-rds`, `syft-enclave`, `syft-bg`, then `syft`) in one go. Always release them together: a partial run can publish a package whose workspace dependency is not on PyPI at the version it needs. You only need to release individual packages if they are changed, but we are not detecting that automatically currently.
3. **Integration tests are optional.** You can skip them during the release if needed. Unit tests should still pass.
4. **Versions are bumped **before releasing to pypi** and pushed automatically** by the release process — no manual version edits required.
5. Merge the release branch back into `dev` (run `uv lock` on that PR and commit `uv.lock`, since the CD bump commits only touch `pyproject.toml`) before cutting any later release, so version bumps and hotfixes are carried forward.

## Hotfixes

If a fix is needed after cutting the release branch, apply the hotfix directly to the release branch and re-release from there.
