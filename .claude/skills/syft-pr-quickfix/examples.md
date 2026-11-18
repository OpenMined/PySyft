# Example findings

Findings the user has judged. `syft-pr-quickfix` reads this file when it is unsure whether a finding
is worth reporting. `syft-pr-quickfix-iterate` adds an entry for every finding the user judges.
Newest last in each section.

## Good — report these

- PR 9571 · `vague-name` · `DatasiteOwnerSyncer._held_before_seed()`
  The name does not say that it returns files. The user had to read the body to learn that it maps
  each peer to the files it holds.
- PR 9571 · `unnamed-value` · `DatasiteOwnerSyncer._held_before_seed()`
  `files.update(dict.fromkeys(unsent, UNSENT_HASH))` builds a dict inline, and the reader cannot
  tell what that dict is. A named variable such as `unsent_files` says it.
- PR 9571 · `name-mismatch` · `DeliveryTracker.clear()` (good, but low priority)
  The docstring says it forgets views and queued sends, but it also forgets accepted changes and the
  seeded flag. A docstring-only fix, so it goes last in the list.
- PR 9571 · `vague-name` · `DatasiteOwnerSyncer.deliver()`
  A bare verb that does not say what it delivers. The skill missed it. The user renamed it to
  `deliver_file_changes()`; `deliver_missing_files()` was rejected because it also sends deletes.
- PR 9571 · `vague-name` · `DatasiteOwnerSyncer._create_current_state_event()`
  "State" alone does not say which state; the codebase has many. The skill missed it. The user
  renamed it to `_create_current_filestate_event()`.
- PR 9571 · `vague-name` · `DeliveryTracker.confirm()`
  A bare verb that does not say what it confirms. The user renamed it to `confirm_deliveries()`.
- PR 9571 · `unlabelled-mapping` · `DatasiteOwnerSyncer._files_held_by_peer_at_seed()`
  Returns `dict[str, dict[str, str]]` without saying what maps to what. Fix: the docstring says
  "Maps each peer email to `{path: hash}`", and `# peer email -> {path: hash}` sits above `held`.

## Bad — do not report these

- PR 9571 · `done-twice` · `SyftboxManager.reject_peer_request()`
  `email_or_peer if isinstance(email_or_peer, str) else email_or_peer.email` appears in two
  methods. A repeated single line is not worth a finding.
- PR 9571 · `done-twice` · `DatasiteOwnerSyncer._seed_views_if_missing()` and `_accepted_changes()`
  Both build the approved emails from `peer_store.approved_peers` in one line. A repeated single
  line is not worth a finding.
