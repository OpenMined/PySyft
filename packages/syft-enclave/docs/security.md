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
running here."_ The enclave generates a fresh encryption and signing keypair and **binds its public keys to that
attestation report**, then shares the report on Google Drive with all peers. How the binding is
achieved differs per target — section 6 — but the effect is the same: the keys cannot be swapped
without the report failing to verify.

Because any peer can **verify the attestation report**, they know those public keys were genuinely
produced by an enclave running the expected open-source container — not by some person who happens to
have access to the account.

> **Status.** This binding is implemented on both targets, by different routes: **Tinfoil** binds
> over a connection its report vouches for (§6.1), **Confidential Spaces** commits to the facts
> inside the signed token itself (§6.2). The Confidential Space route reaches further, covering the
> enclave's email and configured data owners as well as its keys.

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

Section 5 describes the intent. Both targets deliver the first half of it — proof of _what code is
running_ — but they differ on the second half, whether the enclave's keys are bound to that proof.
So this section is split by target.

**Common to both.** A verified report proves the workload runs on genuine confidential-computing
hardware with debug disabled, and that the code and configuration are the ones published: on
Tinfoil, the CVM image and `tinfoil-config.yml` of a named, Sigstore-signed release of the config
repo, including the container image digest it pins; on Confidential Spaces, the image digest in the
token's claims. Neither proves anything about values supplied at deploy time — the enclave's email
and its configured data owners are deploy-time inputs on both targets, and attested code faithfully
relays whatever its deployer handed it.

### 6.1 Tinfoil: the keys come down a connection the report vouches for

A workload cannot inject a nonce into a Tinfoil report — its 64 bytes of user data are the sha256 of
the shim's TLS public key followed by its HPKE public key. But that is exactly what makes binding
possible, because the report **commits to the key terminating a TLS connection to the enclave**. So
the client:

1. verifies the report;
2. opens HTTPS to the enclave and checks the certificate it is served carries that same key;
3. sends a random nonce and checks the enclave signed it with the identity key from the bundle it
   served;
4. and then trusts that bundle, which came down the same connection.

No certificate authority is involved anywhere: the enclave's certificate is self-signed, and the
_report_ is what decides whether to trust it. Step 3 adds what the report cannot carry — that the
enclave **holds the private half** of the key we are about to encrypt to, and that the answer was
produced for this exchange rather than replayed. `attest_peer` then sets those keys for the peer, so
the enclave's public keys are no longer an unsigned Drive file. The bundle is adopted only when both
the pin and the nonce check pass.

Replay is ruled out as a side effect: a captured report commits to a TLS key whose private half
lives in an enclave the attacker does not control, so the pin fails.

The cost is that this needs the enclave **online**. Evidence is still published to
`SYFT_version.json` as provenance, but it is never appraised in place of the live exchange — doing
so would silently mean unbound keys and a replayable report.

### 6.2 Confidential Spaces: the facts are committed to inside the token

Confidential Space gives a workload something Tinfoil does not: it can ask the launcher to embed
bytes of its choosing into the Google-signed token, via `eat_nonce`. Only code running inside the
measured container can do that, and the token's signature is unforgeable. Note that it is the
_signature_ doing the work, not secrecy — measurements and image digests are public on both targets,
deliberately, because that is what makes them auditable.

So the enclave publishes a **claims document** — its email, its configured data owners, its syft
version and its public key bundle — and commits to that document's sha256 in the token. A verifier
recomputes the digest from the published document and compares. The document travels in the clear
and is untrusted; the digest inside the signed token is what makes it true.

There is room for exactly one digest. Slot 0 carries the syft version in plain text, and a nonce is
capped at 74 characters matching `[a-zA-Z0-9_.-]` — which rules out carrying an email address
literally, and leaves a 64-character sha256 hex comfortably inside. Hence one digest over one
document rather than a field per fact.

This closes the gap section 5 describes, and closes it more cheaply than Tinfoil does: the binding
travels **inside the file**, so it needs no connection to the enclave, which suits the offline-first
transport. It also reaches further. Tinfoil binds the enclave's keys; here the same commitment
covers the enclave's **email and its configured data owners**, which are deploy-time inputs and
otherwise unverifiable. That matters most for the data owners, because that list is the approval gate
— a job runs only once all of them approve, so an operator who could change it unobserved could
approve work on their own.

Binding proves the enclave really was started with those values. Whether they are the _right_ values
is the verifier's call, so `AppraisalPolicy` takes an optional `expected_email` and
`expected_data_owners`: left unset the attested values are reported, set they are required.

What it does not give is freshness. The token is minted once at boot and written to
`SYFT_version.json`, so a captured one stays acceptable for the length of the expiry grace window;
the second nonce slot is spent on the claims digest, so there is no room for a per-verifier
challenge as well. Tinfoil gets freshness from its live connection instead. Bounded staleness —
re-publishing periodically and narrowing the grace window — is the remaining work, tracked as H16 in
the [protocol security review](../../../research/protocol-security-review/SUMMARY.md).

For how to deploy either target, see [Confidential Spaces Deployment](./terraform_cs.md) and
[Tinfoil Deployment](./tinfoil_deployment.md).
