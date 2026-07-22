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
permissions for each peer**, which decide who is allowed to see or change it. `syft-client`
expresses those permissions in small **permission files** with a `.gitignore`-like syntax — patterns that say who
can read or write which paths. Most of the time you do not edit them by hand: higher-level
components (the enclave package, the job package) manage them for you, though a user can always set
them directly.

`syft-client` uses the permissions to decide **which files to share with which
peers over Google Drive** — only the files a peer is allowed to read are ever synced to them.

See the [permissions guide](../../syft-permissions/docs/permission-user-docs.md) for the full
syntax.

## 2. Peer-to-peer file sharing

Communication between parties is **transport-agnostic** — it works over any mechanism that can
deliver a file. Today that transport is **Google Drive**. `syft-client` makes "requests" simply by
uploading and downloading shared files: to files with another party, it creates a folder
that both parties can access and uploads the file into it. That is the whole channel — no dedicated
servers, just shared files.

See the [Google Drive connection doc](../../../docs/connections.md) for the folder layout and request flow.

## 3. All files are encrypted and signed

Every file `syft-client` exchanges is **encrypted and signed**. The signature lets the recipient
**cryptographically prove who** produced a file. Encryption means that even if a file is accidentally shared with the wrong person, it is
**unreadable** to them. The shared Google Drive folder is just a delivery mechanism; the security
does not depend on Google Drive keeping anything secret.

## 4. The enclave runs in a Trusted Execution Environment

The computation itself runs inside a **secure enclave**: a **docker container with `syft-client`**
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
if a real human controls a Google Drive account, `syft-client` can reasonably assume that what comes
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
have access to the account. The data owners and the DS then download the enclave's verified keys (and
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
