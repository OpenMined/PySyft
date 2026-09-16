# Security Overview

> **Read this first:** This document assumes you understand the
> [Enclave Flow](./flow.md). That document describes the parties involved and _what_ happens end to end —
> the data owners, the data scientist, the enclave, and how the analysis is submitted and executed. This
> document explains _how_ each of those steps is made secure.

> This is an early **alpha** release — a deliberate zero-to-one effort. The goal
> right now is not scale; it is to push the _Overton window_ of what private collaboration is allowed
> to look like: to demonstrate that two organizations and an outside analyst really can run a joint
> computation where nobody hands over their secrets. Everything is built around that
> **mutual-secrecy** guarantee.

The flow rests on a few security building blocks. This document describes what makes it trustworthy.

## 1. SyftBox and permissions

The basic unit is a **SyftBox**: a local folder of files. Every file carries **read and write
permissions for each peer**, which decide who is allowed to see or change it. `syft`
expresses those permissions in small **permission files** with a `.gitignore`-like syntax — patterns that say who
can read or write which paths. Most of the time you do not edit them by hand: higher-level
components (the enclave package, the job package) manage them for you, though a user can always set
them directly.

`syft` uses the permissions to decide **which files to share with which
peers over Google Drive** — only the files a peer is allowed to read are ever synced to them.

See the [permissions guide](../../syft-permissions/docs/permission-user-docs.md) for the full
syntax.

## 2. Peer-to-peer file sharing

Communication between parties is **transport-agnostic** — it works over any mechanism that can
deliver a file. Today that transport is **Google Drive**. `syft` makes "requests" simply by
uploading and downloading shared files: to files with another party, it creates a folder
that both parties can access and uploads the file into it. That is the whole channel — no dedicated
servers, just shared files.

See the [Google Drive connection doc](../../../docs/connections.md) for the folder layout and request flow.

## 3. All files are encrypted and signed

Every file `syft` exchanges is **encrypted and signed**. The signature lets the recipient
**cryptographically prove who** produced a file. Encryption means that even if a file is accidentally shared with the wrong person, it is
**unreadable** to them. The shared Google Drive folder is just a delivery mechanism; the security
does not depend on Google Drive keeping anything secret.

## 4. The enclave runs in a Trusted Execution Environment

The computation itself runs inside a **secure enclave**: a **docker container with `syft`**
executing in a **Trusted Execution Environment (TEE)** on confidential-compute hardware. The TEE
isolates the running code and encrypts its memory, so the workload is **not controlled or observable
by any person** — not the cloud operator, not the data owners, and not the data scientist. Nobody
can log into it, read its memory, or change what it does.

Inside this environment the enclave enforces the core consent rule: it **only runs computations that
every data owner has approved**. An analysis that touches two owners' data does not execute until
**both** have signed off (see Step 3 of the [Enclave Flow](./flow.md)). The result is a neutral
party that no human steers, yet that runs only what all data owners agreed to.

Note that the security guarantees above cover **where** the code runs and **what data it can
access** — they do not vouch for **what the code does**. Responsibility for the safety of the analysis itself lies
with **both data owners**: each must **read the submitted code carefully** before approving it and
satisfy themselves that it uses the data only for the agreed-upon purpose. Approval is a
deliberate review step, not a formality — the enclave will run exactly what both owners signed
off on, and nothing more.

For how the enclave is built and deployed, see the
[enclave architecture doc](./enclave_architecture.md).

## 5. Attestation bootstraps the encryption handshake

Encryption and signing only help if you know whose keys to trust. For a **person**, this is relatively trivial:
if a real human controls a Google Drive account, `syft` can reasonably assume that what comes
from that account came from them — trust is bootstrapped from the fact that the account is human-operated.

An **enclave is different on purpose**: it must _not_ be controllable by any single person — that is
the entire point of using one. And no one can fully guarantee that no individual has access to the
enclave's Google Drive account, so that account cannot be trusted the way a person's is.

To get around this, the enclave builds a **secure channel out of the insecure account** using **attestation**.
An attestation report is a cryptographically signed statement from the confidential-compute hardware
that says, in effect, _"this exact, open-source docker container (with syft inside it) is what is
running here."_ The enclave generates a fresh encryption and signing keypair, **embeds its public
keys into that attestation report**, and shares the report on Google Drive with all peers.

Because any peer can **verify the attestation report**, they know those public keys were genuinely
produced by an enclave running the expected open-source container — not by some person who happens to
have access to the account.

> **Status on each deployment target.** On **Tinfoil** this binding is implemented, by a different
> route than nonces: the report commits to the TLS key of the enclave's own endpoint, so a peer that
> pins its connection to that key can trust the key bundle served over it. `attest_peer` does this
> and sets the peer's keys from the result. On **Confidential Spaces** it is still unimplemented —
> the channel exists (a workload can inject nonces into the token) but is unused, so there the key
> bundle remains an unsigned Drive file. Section 6 has the detail.

The data owners and the DS then download the enclave's verified keys (and
share their own), and from that point on there is a **trusted, end-to-end secure channel** between the
enclave and every participant.

Crucially, Google Drive is treated purely as an **untrusted transport** — a message-passing channel
and nothing more. The threat model assumes a fully adversarial transport: an attacker (or Google
itself) is presumed able to **read, drop, replay, reorder, or tamper with** anything stored there.
The system does not rely on Google Drive for confidentiality, integrity, or authenticity. Those
properties are enforced end to end by the layers above it — **encryption** protects confidentiality,
**signatures** provide integrity and authenticity (so any tampering is detected and rejected), and
**attestation** anchors the trust to a known enclave. Compromising the transport therefore lets an
adversary at most cause a denial of service; it never yields access to plaintext or the ability to
forge an accepted message.

This is what lets Steps 1–5 of the [Enclave Flow](./flow.md) happen without
anyone trusting Google Drive, the network, or each other.

## 6. What attestation proves on each target

Section 5 describes the intent. What each deployment target actually delivers differs, so this is
the honest accounting. It is written for **Tinfoil**; where Confidential Spaces differs, it is
called out.

**Proves.** The enclave is genuine SEV-SNP/TDX hardware with debug disabled and its firmware TCB at or above minimum; it booted the exact CVM image and config published in a named release of the config repo; that release was signed by GitHub OIDC from that repo's tag via Sigstore; and the container image digest matches the one you pinned.

**Does not prove.** Which email or data owners the enclave was started with. Those are deploy-time `--variable`s, so they are outside the measurement and the attested code merely relays whatever its deployer handed it. Confidential Spaces is in the same position today (`tee-env-*` metadata is not checked either).

**Key binding, and how it is achieved.** A workload cannot inject a nonce into the report: its 64 bytes of user data are the sha256 of the shim's TLS public key followed by its HPKE public key. But that is exactly what makes binding possible — the report _commits to the key terminating a TLS connection to the enclave_. So the client:

1. verifies the report,
2. opens HTTPS to the enclave and checks the certificate it is served carries that same key,
3. checks the enclave signed the client's nonce with the key bundle it served,
4. and then trusts that bundle, which came down the same connection.

No certificate authority is involved anywhere: the enclave's certificate is self-signed, and the _report_ is what decides whether to trust it. `attest_peer` then sets those keys for the peer, so the enclave's public keys are no longer an unsigned Drive file. This is the binding section 5 describes.

It also gets freshness for free: a replayed report commits to a TLS key whose private half lives in an enclave the attacker does not control, so the pin fails.

**Freshness comes from a nonce, not from the report.** A workload cannot influence the report's user data, so the client sends a random nonce and the enclave signs it with the identity key from the bundle it just served. That proves two things the report cannot: the enclave _holds the private half_ of the key we are about to encrypt to, and the answer was produced for _this_ exchange rather than replayed. The bundle is adopted only when both `key_binding` and `nonce_freshness` pass.

**Drive is not a fallback.** Evidence is still published to `SYFT_version.json` — as provenance, and so the path exists if it is ever needed again — but the client always appraises a Tinfoil enclave from the live API. An unreachable enclave is an error, not a downgrade: accepting the Drive copy would silently mean unbound keys and a replayable report.

**On Confidential Spaces**, none of the binding above is implemented. The report is a Google-signed
JWT fetched at boot and written once to `SYFT_version.json`, and `build_eat_nonce()` is called with
no caller nonce — so the enclave's public keys are not committed into it, and a captured token stays
acceptable for the length of the expiry grace window. The channel for fixing this exists (a workload
_can_ inject nonces into the token) but is unused.

For how to deploy either target, see [Confidential Spaces Deployment](./terraform_cs.md) and
[Tinfoil Deployment](./tinfoil_deployment.md).
