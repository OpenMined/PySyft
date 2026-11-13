---
paths:
  - 'packages/syft-enclave/**'
---

# Enclave invariants

Facts that every change to the enclave keeps true. A change that has to break one updates this file
in the same pull request and says why. `candidate` means not yet confirmed by the team. The
`syft-pr-feedback` skill checks pull requests against these.

- **ENC-1 · An enclave runs with encryption required.** `just start` and the Tinfoil deployment set
  `SYFT_ENCLAVE_USE_ENCRYPTION=true`, and `SyftEnclaveClient` logs in with `encryption=True`. Only
  `just start-debug` turns it off. No code path or test relies on an enclave without encryption.
  _(candidate · Koen, PR 9562)_
- **ENC-2 · An enclave runs the syft version of its clients.** Slot 0 of the attestation nonce is
  `syft-<version>`, and a client rejects any other version (`attestation/confidential_space.py`,
  `expected_syft_version` in `attestation/tinfoil.py`). An image is therefore built from the release
  tag of that version. _(candidate · docs/release.md)_
- **ENC-3 · Nonce slots have fixed meanings.** Slot 0 is the syft version and slot 1 is the digest of
  the enclave's claims (`evidence/tee_token.py`). Only the runner mints tokens, so a caller never
  chooses what goes into a slot. _(candidate · Pedro, PR 9523)_
- **ENC-4 · A party's results hold only what that party may see.** A results file written for one
  party does not copy in data another party provided, such as the evaluation prompts.
  _(candidate · Koen, PR 9555)_
