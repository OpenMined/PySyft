# Release Process

## Overview

Releases are managed through dedicated release branches. The mono repo release job handles publishing, tagging and bumping versions for all individual packages automatically.

## Version order

A release publishes the version that is **already on the branch**. The release then tags that version. After the tag, the release job bumps the version for the next release.

The version on a branch is always a version that is **not yet published**. Therefore one version string always refers to one build.

Do not change a version by hand before a release. The release job makes the bump.

## Before the first release from a machine or a new package

Each package uploads with its own PyPI token, stored as a repository secret: `PYPI_PASS_SYFT`, `PYPI_PASS_SYFT_RESTRICT`, `PYPI_PASS_SYFT_PERMISSIONS`, `PYPI_PASS_SYFT_PERMS`, `PYPI_PASS_SYFT_MIGRATION`, `PYPI_PASS_SYFT_DATASET`, `PYPI_PASS_SYFT_JOB`, `PYPI_PASS_SYFT_ENCLAVE` and `PYPI_PASS_SYFT_BG`. `syft-rds` uses trusted publishing instead: its PyPI trusted publisher names the workflow `cd-syft-rds.yml`, and the job runs in the `pypi` environment. Check that every secret exists (`gh secret list`); a missing one fails that package's upload.

## Steps

1. **Bring the release branch up to date with `dev`.** Release branches drop the patch number (`release/v0.10`), so every patch of a minor version is released from the same branch. `dev` is the default branch; `main` is the frozen legacy-PySyft branch.
   ```bash
   git fetch origin
   # Creates the branch for a new minor version, or fast-forwards it for a patch.
   git push origin origin/dev:refs/heads/release/vX.Y
   ```
   The push is refused if the branch has commits that `dev` lacks, such as a hotfix that was not merged back; merge `dev` into the branch in a PR instead. The release runs the workflows of the release branch, so an outdated branch runs outdated workflows.
2. **Run `CD - Monorepo` on the release branch**, from the Actions tab or with `gh workflow run cd-monorepo.yml --ref release/vX.Y -f bump_type=patch`. The run refuses any branch that is not `release/*`. It releases every package one after another, in dependency order: `syft-restrict`, `syft-permissions`, `syft-perms`, `syft-migration`, `syft-dataset`, `syft-job`, `syft`, `syft-rds`, `syft-enclave`, `syft-bg`. Release them together: a package released on its own can name a dependency version that is not on PyPI. Do not push to the release branch while the run is going: each job builds what the branch holds when the job starts, and a push that lands while a job runs makes that job's own push fail.
3. **Tests.** `run_tests` switches lint, unit and integration tests together. Turn it off only for a commit that already passed CI.
4. **The run publishes, tags, then bumps each package.** Do not edit a version by hand. Each job pushes its bump commit and its `<package>/vX.Y.Z` tag (`vX.Y.Z` for `syft`) together.
5. **If a job fails**, fix the cause and use **Re-run failed jobs** on the same run. A job whose version is already on PyPI skips the upload and goes on to tag and bump, so a re-run finishes a release that failed after its upload. Do not start a new run instead: the packages that already released were bumped, so a new run would release them again at the next version. If you must start a new run, deselect the packages that already released.
6. **After the run:**
   - Merge the release branch into `dev` in a PR. Run `uv lock` on that PR and commit `uv.lock`; the bump commits change only `pyproject.toml` files.
   - Create the GitHub Release: `gh release create vX.Y.Z --verify-tag --generate-notes --notes-start-tag v<previous>`.
   - Rebuild the enclave image from the release tag, not from the branch, which already holds the next version. A verifier expects the enclave to run its own syft version, so clients of the new release reject an older image. From `packages/syft-enclave`, on a checkout of `vX.Y.Z`: `just build-push "X.Y.Z latest"` (needs `OM_DOCKER_LOGIN` and `OM_DOCKER_PW`), then update the image digest where the enclave is deployed (`just tinfoil-release X.Y.Z` for Tinfoil, `image_digest` in the Terraform variables for Confidential Space).
   - Add the release to `CHANGELOG.md`.

## Release artifacts

> ⚠️ **Warning: nothing is frozen before 1.0.0.** Until the 1.0.0 release, protocol 0 stays open and is the only protocol of `syft`, `syft-job` and `syft-dataset`, so a change to a version-0 object needs no new object version and no protocol bump. A release that froze protocol 0 would turn every later change into a bump to protocol 1, even before 1.0.0. The release job therefore writes no release artifacts and no release fixtures, and the export scripts that wrote the artifacts are removed. Do not freeze protocol 0, by hand or in CD, until the team decides to for 1.0.0.

A release artifact records the object versions of one release, and the exact schema of each object version. The drift check compares the current models against the artifacts, so an object version cannot change after its release. A release fixture is data exactly as one release wrote it, and later releases test that they still read it (`test_older_protocol_compatibility.py`). Before 1.0.0 there are neither.

To freeze protocol 0 for 1.0.0:

1. Restore the three `export_release_artifact.py` scripts (in `scripts/`, `packages/syft-job/scripts/` and `packages/syft-datasets/scripts/`), the `export-release-artifacts` recipe of the `Justfile`, and the "Freeze release artifacts" step of `cd-monorepo.yml`, `cd-syft-job.yml` and `cd-syft-dataset.yml`. `git log --diff-filter=D --oneline -- scripts/export_release_artifact.py` names the commit that removed them; restore them from its parent.
2. From that release on, generate the release fixtures after each release, each at its own tag, and commit them to `dev` in a PR: `scripts/generate_release_fixture.py` at `vX.Y.Z`, `packages/syft-job/scripts/generate_release_fixture.py` at `syft-job/vX.Y.Z`, `packages/syft-datasets/scripts/generate_release_fixture.py` at `syft-dataset/vX.Y.Z`. Do not run all three at `vX.Y.Z`: syft-job and syft-dataset were bumped earlier in the same run, so at that tag they name the next version.

## Hotfixes

If a fix is needed after cutting the release branch, apply the hotfix directly to the release branch and re-release from there.
