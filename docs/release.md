# Release Process

## Overview

Releases are managed through dedicated release branches. The mono repo release job handles publishing, tagging and bumping versions for all individual packages automatically.

## Version order

A release publishes the version that is **already on the branch**. The release then tags that version. After the tag, the release job bumps the version for the next release.

The version on a branch is always a version that is **not yet published**. Therefore one version string always refers to one build.

Do not change a version by hand before a release. The release job makes the bump.

## Steps

1. **Create a release branch** from `main`, don't include the patch version in the semver, so we can hotfix patches on the same branch (e.g. `release/v0.1`). If you are patching, re-use the branch.
2. **Run the release workflow.** You can trigger frmo github UI from the Actions tab. In most cases, release the mono repo — this releases all individual packages (`syft-client`, `syft-job`, `syft-dataset`, etc.) in one go. You only need to release individual packages if they are changed, but we are not detecting that automatically currently.
3. **Integration tests are optional.** You can skip them during the release if needed. Unit tests should still pass.
4. **The release job publishes, tags, and then bumps the version.** No manual version edit is necessary.
5. Merge the release branch back into `main` to ensure all version bumps and hotfixes are carried forward.

## Release artifacts

`syft-client`, `syft-job`, and `syft-dataset` each write a release artifact. The artifact records the object versions of that release. It also records the exact schema of each object version.

The drift check compares the current models against these files. If an artifact is absent, the drift check has nothing to compare for that version.

The artifacts are inside the package, so the release job runs the export before the build:

```
uv run python scripts/export_release_artifact.py                        # syft-client
uv run python packages/syft-job/scripts/export_release_artifact.py      # syft-job
uv run python packages/syft-datasets/scripts/export_release_artifact.py # syft-dataset
```

A developer can also run an export in a pull request. The version on the branch is the version that the next release publishes. The artifact is therefore available for review before the release.

An artifact is permanent. If an artifact for a version exists, a second export writes nothing and reports success.

An export stops with an error if the protocol changed but the protocol version constant did not change. The error message gives the name of the constant to bump.

The drift check has one known limit. A new protocol generation adds object versions, and no artifact freezes those versions until the release of that generation. The drift check therefore cannot see a change to them. Frequent releases keep this period short.

## Hotfixes

If a fix is needed after cutting the release branch, apply the hotfix directly to the release branch and re-release from there.
