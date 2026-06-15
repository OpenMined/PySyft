# Release Process

## Overview

Releases are managed through dedicated release branches. The mono repo release job handles bumping versions and pushing tags for all individual packages automatically.

## Steps

1. **Create a release branch** from `main`, dont include the patch version in the semver, so we can hotfix patches on the same branch (e.g. `release/v0.1`). If you are patching, re-use the branch
2. **Run the release workflow.** You can trigger frmo github UI from the Actions tab. In most cases, release the mono repo — this releases all individual packages (`syft-client`, `syft-job`, `syft-dataset`, etc.) in one go. You only need to release individual packages if they are changed, but we are not detecting that automatically currently.
3. **Integration tests are optional.** You can skip them during the release if needed. Unit tests should still pass.
4. **Versions are bumped **before releasing to pypi** and pushed automatically** by the release process — no manual version edits required.
5. Merge the release branch back into `main` to ensure all version bumps and hotfixes are carried forward.

## Hotfixes

If a fix is needed after cutting the release branch, apply the hotfix directly to the release branch and re-release from there.
