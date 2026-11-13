# Example findings

Findings the team has judged. `syft-pr-feedback` reads this file when it is unsure whether a finding
is worth reporting, and adds an entry for every finding the user judges. Newest last in each
section. `.claude/skills/syft-pr-quickfix/examples.md` holds more, for names.

## Good — report these

- PR 9604 · `D3` · `disclosures.json` next to the permission files
  The file repeats who may read, which the permission files already say, so the two can drift.
  The author removed it and kept read permissions in the job state.
- PR 9562 · `E1` · `DatasiteOwnerSyncer._send_outbox_message()`
  Returns `False` when the message must wait for encryption. The reviewer asked "Can we not just
  raise?", and the queueing it fed was removed.
- PR 9438 · `D5` · `SyftRDSClient` reads `_sync.datasite_owner_syncer._any_shared_collections`
  A private attribute of another object. The fix added a public property on the syncer.
- PR 9520 · `E5` · the key file written with `write_text()`
  A crash during the write leaves the private keys unreadable. The fix used a lock, a temporary
  file and `os.replace()`.
- PR 9529 · `T1` · the streaming timeout test
  It failed with and without the fix, because the job was killed during setup. The rewrite calls
  `SyftJobRunner._execute_job_streaming()` directly and fails only without the fix.

## Bad — do not report these

- A repeated single line, such as an `isinstance(x, str)` check written in two methods. Not worth a
  helper (from `syft-pr-quickfix/examples.md`).
- A style point that ruff or pre-commit reports.
- "Consider adding more tests" with no named case.
- A problem in code the pull request did not touch.
