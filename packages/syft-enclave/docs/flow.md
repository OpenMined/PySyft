# The Enclave Flow

This document walks through an end-to-end private collaboration in `syft-enclave`, step by step. It
is the best place to start — the [Security Overview](./security.md) builds directly on this flow and
assumes you have read it first.

Users connect via **Jupyter or Google Colab** and sync through **Google Drive**. Using this as a
communication platform, several parties can compute on each other's data. Backed by **confidential
compute**, a **secure enclave** running an **open-source docker container with syft-client** runs
computations from those parties **without any of them having to reveal that data**.

> **Where we are.** This is an early **alpha** release — a deliberate zero-to-one effort. The goal
> right now is not scale; it is to push the _Overton window_ of what private collaboration is allowed
> to look like: to demonstrate that two organizations and an outside analyst really can run a joint
> computation where nobody hands over their secrets. Everything is built around that
> **mutual-secrecy** guarantee.

---

## The parties

Throughout the story there are three parties:

- **DO1** — a data owner who holds a private **benchmark** (for example, loaded in Google Colab).
- **DO2** — a data owner who holds a private **model and its inference code** (also, say, in Colab).
- **DS** — a data scientist who writes the **analysis** that brings the two together.

In the middle sits a **secure enclave**: a sealed, confidential-compute environment that nobody
logs into and that runs a known, open-source docker container.

## Step 0 — Everyone installs PySyft

All parties install the same open-source client (PySyft). DO1 brings a private benchmark, DO2
brings a private model and inference code, and the DS will bring the analysis. The secure enclave
is the neutral ground where the computation will eventually happen — none of the three parties
controls it.

<img src="assets/security/step0.png" width="600" alt="Step 0: all parties install pysyft">

## Step 1 — Data owners upload their data

Each data owner uploads their private asset toward the enclave (the bytes travel through Google
Drive). The data is only ever used inside the enclave: **it never leaks to the other parties, it is
removed after the computation, and an owner can delete it at any time.**

<img src="assets/security/step1.png" width="600" alt="Step 1: data owners upload data">

## Step 2 — The data scientist uploads a private-private analysis

The DS writes the analysis — the code that loads the benchmark, runs it against the model, and
scores the result — and uploads it. The enclave forwards this code to **both** data owners so they
can see exactly what is being proposed against their data before anything runs.

<img src="assets/security/step2.png" width="600" alt="Step 2: upload private-private analysis">

The analysis is just ordinary code. In pseudocode it reads the benchmark, runs the model's
inference on it, and reports a score:

<img src="assets/security/step2a.png" width="600" alt="Step 2: analysis pseudocode">

## Step 3 — Data owners approve the analysis

Because this analysis touches data from two different owners, it only runs once **both** owners
have approved it. Either owner can decline. Nobody — not the DS, not the other owner, not the
operators of the enclave — can force a computation to run over data without that data owner's
explicit consent.

<img src="assets/security/step3.png" width="600" alt="Step 3: data owners approve the analysis">

## Step 4 — The enclave executes the analysis

With both approvals in hand, the enclave runs the analysis inside its open-source docker container
on confidential-compute hardware. The two private inputs meet only here, inside the sealed
environment — never on anyone's laptop and never in plaintext on Google Drive.

<img src="assets/security/step4.png" width="600" alt="Step 4: analysis is executed in the enclave">

## Step 5 — The result is shared with the DS

Only the agreed-upon output leaves the enclave and is delivered to the DS. The private benchmark
and the private model stay where they started. Everyone got something — a result — and nobody had
to give up their secret to get it.

<img src="assets/security/step5.png" width="600" alt="Step 5: result is shared with the data scientist">

---

Next, read the [Security Overview](./security.md) to understand how each of these steps is made
secure.
