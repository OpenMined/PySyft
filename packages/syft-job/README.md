# syft-job

Job submission and execution for SyftBox: data scientists submit bash/Python jobs
into a data owner's inbox, the data owner reviews, approves, and runs them.

## Releasing

Before 1.0.0 a release freezes nothing: protocol 0 stays open, so a change to a
version-0 job object needs no new version and no bump of `JOB_PROTOCOL_VERSION`
(see "Release artifacts" in `docs/release.md`).

From 1.0.0 on, each release freezes its object versions in
`src/syft_job/migrations/history/`, and `scripts/generate_release_fixture.py`
writes a SyftBox tree to `tests/migrations/p2p/fixtures/` exactly as the release
wrote it. Later releases test that they still read those fixtures
(`test_older_protocol_compatibility.py`), and a released object version then
changes only through a new version and a migration.
