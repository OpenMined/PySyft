---
paths:
  - 'syft/sync/peers/**'
  - 'syft/sync/version/**'
  - 'syft/sync/connections/**'
  - 'syft/sync/encryption_policy.py'
  - 'syft/sync/login_utils.py'
---

# Peer and encryption invariants

Facts that every change to peers, peer info and encryption keeps true. A change that has to break
one updates this file in the same pull request and says why. `candidate` means not yet confirmed by
the team. The `syft-pr-feedback` skill checks pull requests against these.

- **PEER-1 · What we know about a peer lives on its `Peer`, in the `PeerStore`.** `Peer.info` holds
  the peer's published info and `Peer.state_at_start` the state at startup. `PeerManager` keeps no
  per-peer dicts beside them. _(candidate · Koen, PR 9562)_
- **PEER-2 · A peer's version, protocols and encryption policy come only from its peer info file**,
  `SYFT_version.json` (`PeerInfo`). A folder name on Drive is an address, not a second source.
  _(candidate · Koen and Pedro, PR 9487)_
- **PEER-3 · A pair encrypts when either side requires it** (`PeerStore.pair_encrypts`). A peer whose
  policy is unknown counts as not requiring it. _(candidate · Koen, PR 9562)_
- **PEER-4 · A SyftBox has one encryption policy**, stored in
  `<syftbox>/<email>/private/syft_settings.json`. Asking for another policy raises
  `EncryptionPolicyMismatchError`. To change the policy, start a new SyftBox.
  _(candidate · Koen, PR 9562)_
