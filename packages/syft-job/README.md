# syft-job

Job submission and execution for SyftBox: data scientists submit bash/Python jobs
into a data owner's inbox, the data owner reviews, approves, and runs them.

## Releasing

On **every** release, export the release artifacts:

```bash
uv run python -m syft_job.migrations.export_release_artifact
```

This always writes `src/syft_job/migrations/history/syft-job-<version>.json`
(the package's identity + the protocol it speaks), and additionally writes
`protocol-<n>.json` when the release ships a new protocol version. The script
refuses to run if the job protocol changed without bumping
`JOB_PROTOCOL_VERSION` (`src/syft_job/migrations/registry.py`).

Tests check the code against these artifacts: released object versions are
frozen forever — changing one requires a new version plus migrations.
