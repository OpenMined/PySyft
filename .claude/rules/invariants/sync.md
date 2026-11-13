---
paths:
  - 'syft/sync/sync/**'
  - 'syft/sync/syftbox_manager.py'
---

# Sync invariants

Facts that every change to syncing and delivery keeps true. A change that has to break one updates
this file in the same pull request and says why. `candidate` means not yet confirmed by the team.
The `syft-pr-feedback` skill checks pull requests against these.

- **SYNC-1 · An owner's sync records its local changes, pulls each peer's proposals, then delivers
  once** (`DatasiteOwnerSyncer.sync()`). Local changes come first, so a proposal based on an older
  copy of a file cannot overwrite a newer edit by the owner. _(candidate · Koen, PR 9571)_
- **SYNC-2 · Every peer that may read a file gets its changes**, including the peer whose proposal
  was accepted. Delivery planning asks `perm_context.can_read()`; there is no second send path for
  accepted changes. _(candidate · Koen, PR 9571)_
- **SYNC-3 · One failure does not stop the rest.** A proposal that fails is logged and the sync goes
  on. A send that fails is tried at most `MAX_SEND_ATTEMPTS` times in the same pass, then dropped
  with an error in the log. _(candidate · Koen, PR 9571)_
