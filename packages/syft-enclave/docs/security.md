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

> **Status.** Implemented on both targets, binding the same facts — keys, email and configured data
> owners — by different mechanisms: **Confidential Spaces** commits to a digest of them inside the
> signed token (§6.1), **Tinfoil** signs them with a key its report vouches for, over a connection
> pinned to that key (§6.2). Section 6 sets out what each one proves, and §6.3 compares them.

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

Section 5 says the enclave binds its own public keys to its attestation report, so that nobody can
swap those keys for their own. Confidential Spaces and Tinfoil both do that, by different means.
This section sets out what each one proves about an enclave, and what each one leaves open.

### 6.0 What has to be bound, and why

A verified report proves the workload runs on genuine confidential-computing hardware with debug
disabled, and that the **code and configuration** are the ones published. On Tinfoil that means the
CVM image and the `tinfoil-config.yml` of a named Sigstore-signed release, including the container
image digest that the config pins. On Confidential Spaces it means the image digest in the report's
claims.

A verified report is not enough on its own. Three of the things a peer needs are runtime values,
which sit outside what the report measures:

| Fact                                | Why it matters                                                                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the enclave's **public key bundle** | you are about to encrypt private data to it. Whoever controls the enclave's Drive account can swap an unbound bundle, and then read everything you send. |
| the enclave's **email**             | it identifies the datasite you are talking to.                                                                                                           |
| its configured **data owners**      | this list is the approval gate. A job runs only once _all_ of them approve, so anyone who can change the list unseen can approve work on their own.      |

So the enclave writes those facts into one document — its email, its data owners, its syft version
and its public key bundle — and then binds that document to its attestation report. The document
itself always travels in the clear, and a verifier never trusts the document on its own.
Confidential Spaces and Tinfoil differ only in how each one makes the document trustworthy.

### 6.1 Binding extra facts on Confidential Spaces

On Confidential Spaces the attestation report is a token: a signed statement that Google's launcher
issues to the enclave at boot. The launcher lets the code inside the enclave add a few values of its
own to that token, in a field called `eat_nonce`. Only code inside the measured container can ask
for this, and nobody can forge Google's signature on the result.

The enclave hashes its claims document with sha256, then asks the launcher to put that hash in the
token. A verifier reads the published document, hashes it the same way, and compares the two hashes.
If anyone alters the document, the two hashes no longer match.

The token is public, and we do not try to hide it. Anyone can read the measurements and the image
digest inside the token, which is what lets anyone audit what the enclave runs. Google's signature
is what makes the hash trustworthy, not secrecy.

The `eat_nonce` field holds a short list of values. Slot 0 of that list already holds the syft
version as plain text, which leaves one spare slot, and the launcher caps each value at 74
characters drawn from `[a-zA-Z0-9_.-]`. An email address does not fit, because `@` is not in that
set, while a 64-character sha256 hash does. That is why the enclave binds one hash of one document,
rather than one value per fact. The token carries the hash, so a verifier needs no connection to the
enclave, which suits a transport built out of files.

**Limitation: the enclave never asks for a new token, so a key can never be retired.** The enclave
asks for one token at boot and writes it to `SYFT_version.json`. Google issues these tokens with a
short life, but the verifier accepts a token up to a month old, because the enclave does not yet ask
for a new one. Anyone who keeps a copy of an old token can present it later as though it were
current. Presenting an old, captured token is called a replay.

A replay does not let an attacker read your data. The token binds the enclave's key bundle, so
replaying an old token also replays an old bundle, and the private half of that bundle never left
the enclave that made it. A replayer can make you encrypt data to a key that nobody holds any more,
which stops the work, but cannot read what you send. What a replay does cost you is the ability to
retire a key. Say a past enclave's private key becomes known to an attacker. That enclave's token
stays acceptable for ever, so the attacker can keep presenting the token, and can then decrypt what
you send. Checking that a token is recent is the only way to say "stop trusting that enclave", so
without such a check there is no way to revoke one.

Two things make a replay harder. A replayer has to control the enclave's Drive account, because that
is where a verifier reads the token from. And a token from a debug-mode enclave, where an operator
can log in over SSH and read the key, already fails the `dbgstat` check. A verifier can also refuse
an old token by saying what it expects: `AppraisalPolicy` takes `expected_image_digest` and
`expected_data_owners` and `expected_email`, and a check fails if the enclave runs a different
image, lists different data owners, or runs as a different datasite. A policy refuses to be built
without all three, so a verifier cannot skip those checks by accident. To verify without pinning,
say so with `allow_unpinned=True`.

This is a current limitation, and 6.4 lists the plan for removing it. Two assumptions make the
limitation acceptable for now. The enclave's private key never leaves the enclave, which is a
reasonable thing to rest on, so an attacker holds no old key to pair with a replayed token. And an
enclave is short-lived, so few old tokens exist to replay.

### 6.2 Binding extra facts on Tinfoil

Tinfoil gives the code inside the enclave no way to add anything to the report, so the enclave binds
its claims document a different way: it signs the document, over a connection that the report
vouches for.

All 64 bytes of user data in a Tinfoil report are already in use. Those bytes hold the sha256 of the
shim's TLS public key, followed by the shim's HPKE public key. That is what makes binding possible,
because the report commits to the key that terminates a TLS connection to the enclave. So the client:

1. verifies the report;
2. opens HTTPS to the enclave and checks the certificate it is served carries that same key, which
   proves the connection ends inside the attested enclave;
3. takes the key bundle served over that connection, which is now authentic;
4. sends a random nonce, and checks the enclave signed **the nonce together with the claims
   document** using the identity key from that bundle.

No certificate authority takes part. The enclave's certificate is self-signed, and the report is
what decides whether to trust the key inside that certificate. The signature in step 4 proves three
separate things: the enclave holds the private half of the key you are about to encrypt to, the
answer was produced for this exchange rather than an earlier one, and the claims are the facts the
enclave meant to assert. The client accepts the bundle and the claims only if step 2 and step 4 both
pass.

Tinfoil can retire a key, because every check is live. A captured report commits to a TLS key whose
private half sits in an enclave the attacker does not control, so the certificate check fails, and
the nonce is new on every request.

### 6.3 Side by side

|                                   | Confidential Spaces             | Tinfoil                                            |
| --------------------------------- | ------------------------------- | -------------------------------------------------- |
| hardware, code, config            | ✅                              | ✅                                                 |
| key bundle bound                  | ✅ hash inside the signed token | ✅ served over a connection the report vouches for |
| email and data owners attested    | ✅ same hash                    | ✅ signed with the bound key                       |
| freshness, so keys can be retired | ❌ one token, issued at boot    | ✅ live connection and a per-request nonce         |

Both targets appraise the facts the same way, once the document is trustworthy. `AppraisalPolicy`
for Confidential Spaces and `TinfoilAppraisalPolicy` for Tinfoil take the same expectations and run
the same comparison. Each policy has to pin `expected_image_digest`, `expected_data_owners` and
`expected_email`, or say `allow_unpinned=True`. Binding proves the enclave started with those
values. Whether they are the right values is the verifier's call.

### 6.4 Todo

- **Confidential Spaces: ask for a new token periodically, and narrow the window the verifier
  accepts** (H16 in the
  [protocol security review](../../../research/protocol-security-review/SUMMARY.md)). Google issues
  these tokens with roughly a 30-minute life. `JWT_EXPIRY_GRACE_SECONDS` in the verifier widens that
  to a month, because the enclave writes its token once at boot. Asking for a new token on a timer,
  and cutting the window, bounds how old a token can be. That is enough to revoke one, and it does
  not need the spare nonce slot.

For how to deploy either target, see [Confidential Spaces Deployment](./terraform_cs.md) and
[Tinfoil Deployment](./tinfoil_deployment.md).
