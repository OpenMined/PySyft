---
paths:
  - 'packages/syft-job/**'
---

# Job invariants

Facts that every change to jobs keeps true. A change that has to break one updates this file in the
same pull request and says why. `candidate` means not yet confirmed by the team. The
`syft-pr-feedback` skill checks pull requests against these.

- **JOB-1 · A job runs only the exact submission the data owner approved.** Before a run,
  `SyftJobRunner` compares the hash of the submission with `approved_hash`, and refuses on a
  mismatch. _(candidate · PR 9536)_
- **JOB-2 · A job name never contains `@`.** So an index that contains `@` is always a datasite
  email (`JobStorage.validate_job_name()`). _(candidate · Koen, PR 9497)_
- **JOB-3 · Permission files decide who may read a job's files.** The job state records the
  requested and the approved read permissions (`requested_read_permissions`,
  `approved_read_permissions`). No other file stores who may read. _(candidate · Koen, PR 9604)_
- **JOB-4 · `JobStorage` is the only code that reads or writes job files**, and a protocol codec
  (`protocolcodecs/`) decides where they live on disk. `JobClient` and `SyftJobRunner` never branch
  on a protocol version. _(candidate · Koen and Pedro, PR 9434)_
