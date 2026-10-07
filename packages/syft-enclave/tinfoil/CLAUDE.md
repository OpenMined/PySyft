# Signed receipts on Tinfoil

When a job finishes, the enclave writes `receipt.dsse.json` into the job's outputs. The receipt
records which code ran, on which data, who approved it, what it produced, and where it ran. The
enclave signs it. The code is in `src/syft_enclaves/receipt/`.

## Why a receipt can be trusted

The trust passes through five steps, in this order:

1. The attestation proves that a genuine enclave booted the measured config. That config pins
   the data owners (`SYFT_ENCLAVE_DATA_OWNERS`) and whether receipts are on
   (`SYFT_ENCLAVE_RECEIPTS`), so a verifier checks both against the signed release.
2. The data owners reach the enclave through Google Drive, so each approval comes from that data
   owner's own account.
3. The data owners do not collude. A job runs only after every one of them approves it, so a job
   that runs is not malicious.
4. The enclave signs the receipt with its Ed25519 identity key, and the attestation binds that
   key to the enclave (see below).
5. So a receipt with a valid signature describes a run that the enclave actually did.

## How the key is bound on Tinfoil

There are two bindings: one for a verifier who can reach the enclave now, and one carried in the
receipt.

**Live, over a pinned connection.** The boot report (`/tinfoil/attestation.json`) cannot carry the
identity key: its report data holds the shim's TLS key. So this binding goes through that TLS key.
A verifier fetches the enclave's key bundle and claims over a connection pinned to the TLS key,
and the enclave signs them with the identity key. `attest_peer` returns that bundle as
`verified_key_bundle`, and `verify_receipt` checks the receipt against it.

**In the receipt.** This one can be checked without reaching the enclave. Once per boot, the
enclave asks `/tinfoil/attestation.sock` for a second report whose nonce is the
sha256 of the run key (`receipt/key_binding.py`), and puts it in every receipt as
`execution.attestation.keyBinding`. One report per boot is enough: the run key is made at boot and
signs every receipt after that. This needs CVM image 0.14.12 or later and `attestation: true` on
the container. When the socket is missing, `keyBinding` is null.

`verify_receipt` checks that the report's nonce is the run key's hash and that its report data is
derived from that nonce. It does not yet check the hardware signature on this report: the tinfoil
Python SDK cannot appraise v3 documents, only tinfoil-go can.

## What the receipt holds

The receipt is an [in-toto statement](https://in-toto.io/Statement/v1). The job writes part of it
and the enclave writes the rest.

The job may write `outputs/receipt_claims.json` with these keys, and no others:

- `subject`: what the receipt is about, usually the model, by name and digest
- `model`: the base model, the adapter and the sampling settings
- `eval`: the eval set
- `results`: counts and metrics

Each digest names its scheme: `dirhash-sha256/1` for a directory (sha256 of the canonical JSON of
`{relative path: sha256}`), `file-sha256/1` for a file, `jcs-sha256/1` for JSON (sha256 of its RFC
8785 form), `hf-revision/1` for a Hugging Face commit, `modelwrap/2` for model weights (the
dm-verity root hash of Tinfoil's modelwrap pack, schema 2, which anyone can recompute with
`modelwrap --schema 2 <repo>@<commit>`), and `oci/1` for a container image.

The eval job computes `modelwrap/2` itself, without Docker: it fetches the modelwrap binary and the
pinned `mkfs.erofs` by sha256 and runs `modelwrap --local`. It lands in `model.weights` with role
`base_weights`.

These are only as true as the job's code. Every data owner approved that code, and the receipt
carries all of it. A job that writes no claims file still gets a receipt, without these sections.
A claims file with any other key, or that is not valid JSON, gets `receipt_error.txt` instead.

The enclave writes:

- `job`: the full content of every submitted code file
- `datasets`: the path and sha256 of every private dataset file
- `outputs`: the full content of every output file, except the receipt and the claims file
- `execution`: platform, CVM version, config digest, runtime image digest, start and finish times,
  the run's public key, the Tinfoil attestation document, and the release it was verified against
- `parties`: the submitter, and each data owner with the datasets they gave
- `consent`: who approved and when, and the sha256 of the code they approved
- `policy`: who is sent the results and the receipt

The receipt is a DSSE envelope: the JSON statement plus a signature over it.

## Logging on Rekor

The benchmark owner's notebook uploads the receipt to Rekor, Sigstore's public log, as a `dsse`
entry. The benchmark owner sends it, but the enclave's key signed it, and Rekor checks that
signature. Rekor stores the hashes of the receipt, the signature and the public key, but not the
receipt, so the log shows when the enclave signed the receipt without showing what is in it.

## Two releases

`tinfoil-config.yml` has receipts off and `tinfoil-config-receipts.yml` has receipts on. Both pin
the same image, and each is published under its own tag with `just tinfoil-release`. Releases come
from [`OpenMined/syft-enclave-tinfoil`](https://github.com/OpenMined/syft-enclave-tinfoil), and
the files in this folder are the working copies it copies over. See "Where the configs live" in
`docs/tinfoil_deployment.md`.
