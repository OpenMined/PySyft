# syft-job

Job submission and execution for SyftBox: data scientists submit bash/Python jobs
into a data owner's inbox, the data owner reviews, approves, and runs them.

## Releasing

On **every** release (after bumping the version), run both release scripts:

```bash
uv run python scripts/export_release_artifact.py
uv run python scripts/generate_release_fixture.py
```

`export_release_artifact.py` always writes
`src/syft_job/migrations/history/package-artifacts/syft-job-<version>.json`
(the package's identity + the protocol it speaks), and additionally writes
`history/protocols/protocol-<n>.json` when the release ships a new protocol
version. It refuses to run if the job protocol changed without bumping
`JOB_PROTOCOL_VERSION` (`src/syft_job/migrations/registry.py`).

`generate_release_fixture.py` writes a full SyftBox tree —
`tests/migrations/p2p/fixtures/syft_job-<version>-protocol<p>_syftbox/` — exactly
as this release serializes jobs to disk. Commit it: future releases loop over
these fixtures (`test_older_protocol_compatibility.py`) to prove they can still
read and round-trip older on-disk data.

Tests check the code against these artifacts: released object versions are
frozen forever — changing one requires a new version plus migrations.
