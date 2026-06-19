# V1 Upgradeability — Discovery, Findings & Implementation Strategy

> **Status:** Draft for team alignment
> **Audience:** syft-client engineers
> **Scope:** V1 = upgrades between the current `dev` version and every currently-supported, backward-compatible version (the *patch cohort*). V2 (forced upgrades across minor/major boundaries) is explicitly out of scope, but every V1 decision here is checked against "does this stay elegant for V2?"
> **Companion artifact:** [`phase0_repro.py`](./phase0_repro.py) — a runnable reproduction of the real folder-resolution behaviour quoted throughout this doc. Run it with `uv run python docs/upgrades/phase0_repro.py`.

---

## 0. TL;DR

- **The promise:** a user can upgrade their syft-client and keep working — their local SyftBox assets survive, and peers on nearby versions keep talking. The non-negotiable invariant: **a patch upgrade (e.g. `0.1.111 → 0.1.117`) never loses or strands a user's assets, peer or no peer.**
- **Good news, and a surprise:** the one historical backward-compat break (folder names started embedding the version at `0.1.112`) has **already been hot-patched** by PR #280 (`_filter_patch_compatible`). A solo patch upgrade with a single existing folder resolves correctly today — we proved it (§3, Scenario A).
- **The real finding is not "something is broken" — it's "there is no upgrade *system*, only point-fixes."** #280 is a discovery-time band-aid living inside the Drive transport. It leaves two sharp edges we reproduced live: a **hard `RuntimeError`** when two patch folders coexist (Scenario B), and **silent stranding** the moment a minor bump happens (Scenario C). The same "we patched the seam that happened to break" pattern leaves un-versioned local caches and unverified checkpoint restores as latent traps (§4).
- **The work splits cleanly:** a small amount of **pre-work** (§5) that pays off no matter what, then a **translator + upgrader** design (§6) anchored on the seams that already exist.
- **One decision the team must make now** (§8): keep version-scoped folders and bridge them, or retire version-in-folder-name and move version-awareness into content. Everything downstream depends on it.

---

## 1. The Contract — what "upgrade" actually promises

Before any code, we agree on words. The earlier confusion in planning came from mixing two different things ("translation" vs. "forced upgrade"); this section nails them down.

### 1.1 The surfaces

An upgrade is never just "my binary changed." Three surfaces move at once:

| Surface | What lives there | Examples in code |
|---|---|---|
| **Local** | On-disk state owned by this client | checkpoints, rolling state, caches under `~/.syftbox`, crypto keys |
| **Peer** | Messages exchanged with another client | `ProposedFileChangesMessage`, `FileChangeEventsMessage` |
| **Remote** | Shared transport layout on Google Drive | folder names, `SYFT_version.json`, message files |

**Upgrades are handled per-peer.** Two clients negotiate compatibility pairwise; there is no global "everyone is on version X" assumption (`syft_client/sync/version/peer_manager.py`).

### 1.2 The two upgrade classes

This is the core model. They are different mechanisms, not two flavours of one thing.

- **Patch skew → translation path.** Versions differ only in patch (`0.1.111` vs `0.1.117`; same `major.minor`). Peers keep communicating. The **higher version is the "service-bus manager"**: it knows how to translate down to / read up from the lower version. **No upgrade is forced.**
- **Minor/major skew → forced-upgrade path.** Versions differ in `major` or `minor`. The **older peer must upgrade.** There is **no translation here** — do not build translation into this path.

This maps directly onto the existing compatibility enum (`syft_client/sync/version/version_info.py:50-66`): `SAME` and `PATCH_DIFF` are the translation cohort; `INCOMPATIBLE` is the forced-upgrade cohort.

### 1.3 The non-negotiable invariant

> A local patch upgrade never loses or strands a user's assets — **with or without a peer involved.**

If a user runs `0.1.111`, writes data, and upgrades to `0.1.117`, their assets must still be there and still sync. If serialization changed in the new patch, **the user should never have to know or care.** This is the bar V1 is measured against.

### 1.4 V1 vs V2 boundary

- **V1** = `dev` + all currently-supported backward-compatible versions = the patch cohort (§2.1).
- **V2** = `dev` + versions that require a minor/major (forced) upgrade.
- **Out of scope for V1:** cross-minor migration, any translation on the forced-upgrade path, sub-package format migration (tracked as a risk in §4.4).

---

## 2. Findings — what the codebase does today

### 2.1 The supported set

The single source of truth is `syft_client/version.py`:

```python
SYFT_CLIENT_VERSION = "0.1.117"
MIN_SUPPORTED_SYFT_CLIENT_VERSION = "0.1.93"
PROTOCOL_VERSION = "1.0.0"
```

The entire supported range — **`0.1.93` → `0.1.117`** — sits inside a single minor line (`0.1.x`). By the project's own rule (`compatibility_status_with`, `version_info.py:63`), **every pair in that range is mutually patch-compatible.** So the V1 cohort is, conveniently, *all currently-shipped versions*.

> **Assumption for the team to confirm:** V1 targets `0.1.93 → dev`. `0.1.93` is the natural floor (birth of version negotiation). If we want a *hard* enforced floor, see biting issue §4.1 — it does not exist yet.

### 2.2 The audit matrix

Every serialized surface, audited across the supported range:

| Surface | File | Changed in range? | Backward-compatible? |
|---|---|---|---|
| Proposed-filechange wire msg | `sync/messages/proposed_filechange.py` (`msgv2`) | No | ✅ |
| File-change event wire msg | `sync/events/file_change_event.py` (`syfteventsmessagev3`) | No | ✅ |
| Checkpoint / rolling state | `sync/checkpoints/{checkpoint,rolling_state}.py` | Additive only; `CHECKPOINT_VERSION`/`ROLLING_STATE_VERSION` still `1` | ✅ |
| Version handshake | `sync/version/version_info.py` | +2 `Optional=None` fields | ✅ |
| **GDrive folder naming** | `sync/connections/drive/gdrive_transport.py` | **Yes — broke at `0.1.112`** | ❌ → **hot-patched by #280** |
| Local caches / keys | `persisted_dict.py`, `peer_store.py` | Un-versioned plain JSON | ⚠️ no mechanism |

Two structural facts make the content surfaces forgiving — and dangerous:
- **No model uses `extra="forbid"`** (default is `extra="ignore"`, some `extra="allow"`). Unknown fields don't crash deserialization → great for forward-compat, but a breaking change can **silently drop data** with no error.
- Content (de)serialization funnels through a small number of seams: `as_compressed_data` / `from_compressed_data` on the message models, and **14 (de)serialization call sites in `sync/connections/connection_router.py`**. That concentration is what makes a translator feasible (§6.3).

### 2.3 The one real break — and that it's already patched

At `0.1.112` (commit `59b6208c46`) folder names began embedding the running version. The write path still does this today:

| Folder | Builder | Produces (running `0.1.117`) |
|---|---|---|
| Personal SyftBox | `GdrivePersonalSyftboxFolder.as_string` (`gdrive_transport.py:142`) | `0.1.117#alice@…` |
| P2P inbox/outbox | `GdriveP2PFolder.as_string` (`:117`) | `syft_datasite#0.1.117#alice@…#outbox#bob@…` |
| Checkpoints | `_get_checkpoints_folder_name` (`:1955`) | `alice@…-0.1.117-checkpoints` |

Left alone, that would strand every folder on the first patch upgrade. It doesn't — because PR **#280 ("fix: patch upgrades compatible on older versions")** added a discovery-time bridge:

- `_filter_patch_compatible` (`gdrive_transport.py:224`) keeps folders whose embedded version shares the running `major.minor`.
- `_expect_one` (`:1432`) returns the single survivor, or **raises** if more than one.

So the read path looks **across** patch versions even though the write path stamps an exact version. **That asymmetry is the whole story of V1.**

### 2.4 The philosophical conflict

Version-in-folder-name was built for **isolation** — keep each version in its own silo so they can't corrupt each other. Your feature wants the **opposite**: sharing and translation across versions. #280 is the seam where these two philosophies already collide, and it resolved the collision with a filter rather than a decision. **V1 has to actually make the decision** (§8.1).

---

## 3. Phase-0 reproduction (observed, not predicted)

`phase0_repro.py` drives the *real* functions (`_filter_patch_compatible`, `_expect_one`, the folder builders); only the module-level `SYFT_CLIENT_VERSION` is swapped to play "the binary the user is running." Verbatim output:

**Scenario A — solo patch upgrade, one old folder (the good news / #280 working):**
```
  on Drive : ['0.1.111#alice@openmined.org']
  running  : 0.1.117
  resolved : id::0.1.111#alice@openmined.org
  VERDICT  : FOUND — assets NOT stranded (patch-compat discovery works)
```

**Scenario B — two patch folders coexist (pre-#280 residue, or a straddled upgrade):**
```
  on Drive : ['0.1.111#alice@openmined.org', '0.1.112#alice@openmined.org']
  running  : 0.1.117
  error    : Found 2 compatible folders on Drive: [...]. Exactly one is expected.
             ... Please delete the stale folder(s) on Drive ... and retry.
  VERDICT  : HARD FAIL — RuntimeError, user must hand-delete a folder on Drive
```

**Scenario C — minor bump (the V1/V2 boundary):**
```
  on Drive : ['0.1.117#alice@openmined.org']
  running  : 0.2.0
  resolved : None
  VERDICT  : STRANDED — minor mismatch filtered out (forced-upgrade territory = V2)
```

**Reading of the results:**
1. The headline fear ("patch upgrade strands my data") is **already handled** for the common case. Good.
2. But the fix is brittle at the edges: **Scenario B turns a stale folder into a hard stop that demands manual Drive surgery** — that is not "the user shouldn't have to care." And the write path keeps minting exact-version names, so the conditions for B accumulate over time.
3. **Scenario C is correct behaviour** (it *should* strand → force an upgrade) but today it strands **silently** — no message, no migration offer. That's the V2 hand-off point, and it needs to be explicit, not a void.

---

## 4. The "Biting" issues

Latent risks that won't fail today but will sink V2 if V1 papers over them. Each gets a V1 verdict so nothing is dropped by silence.

### 4.1 `MIN_SUPPORTED_SYFT_CLIENT_VERSION` is declared but never enforced
`version.py:12` defines it; nothing gates on it. It's advertised in `VersionInfo` and ignored. **Bites because:** V2's forced-upgrade path *needs* a real floor to refuse too-old peers cleanly. **V1 verdict:** make it real in pre-work (§5) — cheap now, load-bearing for V2.

### 4.2 Local caches and keys carry no version stamp
`PersistedDict` writes plain `json.dumps` (`persisted_dict.py:111`); `PeerStore` keys are plain JSON (`peer_store.py:235`). No schema marker anywhere. **Bites because:** if a patch ever changes these shapes, there is **nothing to translate from** — you can't migrate what doesn't say what version it is. **V1 verdict:** version-stamp them in pre-work (§5). This is the single highest-leverage prerequisite.

### 4.3 No integrity re-verification on checkpoint/rolling-state restore
Restore trusts stored hashes; nothing re-hashes content against `CheckpointFile.hash` on the way back in. **Bites because:** a migration that rewrites bytes has no safety net — silent corruption looks like success. **V1 verdict:** the upgrader must re-hash on migrate (§6.4); the primitives already exist (`compute_file_hashes`, `file_utils.py:7`).

### 4.4 Sub-packages version independently, pinned `==`
`syft-job`, `syft-dataset`, `syft-permissions`, etc. move on their own tracks. A syft-client patch can pull a sub-package that changed a serialized shape (e.g. a `Job`). **Bites because:** our whole "patch == compatible" assumption is scoped to `syft_client` core; these are an unaudited surface. **V1 verdict:** out of scope for V1 *implementation*, but **must be audited before we lock V1 scope** (decision §8.3).

### 4.5 The `_expect_one` hard-fail (from §3, Scenario B)
Not a separate subsystem — but called out here because it's the one place current behaviour actively violates the invariant. **V1 verdict:** the upgrader subsumes this — on detecting multiple patch-compatible folders it should *reconcile/merge*, not throw.

---

## 5. Pre-work — fixes that strengthen the path regardless of design

These are small, independently-valuable changes that make *any* upgrade strategy tractable. They are not the feature; they're the ground the feature stands on. Each can ship before the §8 decision is made.

| # | Pre-work item | Why it unblocks upgrades | Touches |
|---|---|---|---|
| **P1** | **Stamp a version/schema marker on every persisted artifact.** Add a small header (`{schema, producer_version}`) to the local caches and key stores. `compress_data` already wraps payloads in a tar with a fixed member (`syftbox_utils.py:66`) — add a sidecar member rather than reformatting. | You cannot translate from an artifact that doesn't declare its version. Fixes §4.2. Prerequisite for the entire translator. | `persisted_dict.py`, `peer_store.py`, `syftbox_utils.py` |
| **P2** | **Make `MIN_SUPPORTED_SYFT_CLIENT_VERSION` enforced.** One real gate at login/peer-negotiation. | Gives the forced-upgrade path a clean refusal point. Fixes §4.1. | `login_utils.py`, `peer_manager.py` |
| **P3** | **Add integrity re-verification on checkpoint/rolling-state restore.** Re-hash content vs. stored hash; raise/repair on mismatch. | The safety net every migration needs. Fixes §4.3. | `checkpoints/checkpoint.py`, `connection_router.py` |
| **P4** | **Centralize the (de)serialization seam.** The 14 call sites in `connection_router.py` should route through one read/one write helper. | A translator needs *one* place to hook, not 14. Pure refactor, no behaviour change. | `connection_router.py` |
| **P5** | **Freeze a cross-version corpus now.** Capture serialized bytes for each supported version into `tests/fixtures/` before any of them age out. | The regression guard that proves §2.2 stays true forever (§9). Cheap now, impossible later. | `tests/fixtures/`, new test |
| **P6** | **Make folder resolution merge-tolerant.** Replace the `_expect_one` throw with a reconcile path (pick the data-bearing folder, fold the rest). | Turns Scenario B from a hard stop into a non-event. | `gdrive_transport.py:1432` |

> **Sequencing note:** P1, P4, P5 are prerequisites for the translator; P2, P3, P6 are independent hardening. None require the §8 decision, so they can start immediately and de-risk the estimate.

---

## 6. Technical Design

### 6.1 Principles
1. **Every artifact is self-describing** about its version (delivered by P1).
2. **Migrations are pure functions**, registered by `(artifact_type, version)`, unit-testable in isolation.
3. **One funnel, not scattered patches** — translate at the seam (P4), not at each call site.
4. **Integrity is verified on every migration** (P3), never assumed.

### 6.2 The Version Translator
A registry of pure functions in two directions:

- **Upcast (read-up)** — `(type, from_version) → current model`. Serves the **local self-upgrade** case: open old artifact, normalize to today's shape. The existing `pre_init` `model_validator(mode="before")` hooks (`proposed_filechange.py:51`, `file_change_event.py:93`) are *already* doing exactly this for the legacy `content_type=None` case — they are the idiomatic home for per-model upcasts.
- **Downcast (write-down)** — `current model → (target_peer_version) wire shape`. Serves the **peer-skew** case: the newer "service-bus manager" peer emits the older peer's dialect. Keyed off the peer's advertised `VersionInfo` (already exchanged via `SYFT_version.json`).

```
deserialize(type, raw_bytes, source_version) -> CurrentModel:
    model = lenient_parse(raw_bytes)
    return upcast_chain(type, source_version, model)   # old → … → current

serialize(type, model, target_version) -> bytes:
    return emit(downcast_chain(type, target_version, model))  # current → … → target
```

`source_version` comes from the artifact's own stamp (local) or the peer's `VersionInfo` (peer).

### 6.3 Seams / plug-points
- **Primary:** the centralized read/write helper from P4 in `connection_router.py` — covers wire messages + checkpoint/rolling traffic in one place.
- **Per-model:** the `pre_init` validators — where individual upcast rules live.
- **The one a naive design misses:** folder discovery in `gdrive_transport.py` (`as_string`, `from_name`, `_filter_patch_compatible`). The historical break lives *here*, not at the Pydantic layer. The translator concept must extend to **folder naming**, not just content.
- Plus the direct-deserialize sites that bypass the router (`gdrive_transport.py:616, 1035`).

### 6.4 The Upgrader
A driver that runs on version change and owns **asset cataloguing + integrity**:
1. **Enumerate** assets across local + remote (folders, caches, checkpoints).
2. **Migrate** each via the translator.
3. **Re-hash and verify** (P3) — `compute_file_hashes` (`file_utils.py:7`); refuse to discard the source until the target verifies.
4. **Reconcile** the Scenario-B multi-folder case (subsumes P6).
5. **Write an explicit manifest** (§6.5). Idempotent and resumable — a half-finished upgrade must be safe to re-run.

### 6.5 The Asset Catalog/Manifest
Today the catalog is *implicit*: a set of folders + the `content_hash` baked into dataset folder names (`DatasetCollectionFolder`, `gdrive_transport.py:146`) + the `file_hashes` `PersistedDict`. V1 makes it **explicit**: `{asset_path → content_hash → source_version}`. That's what makes migration **verifiable** (did everything arrive?) and **idempotent** (skip what's already migrated).

### 6.6 The folder-naming decision (forward pointer)
The design above works under *either* answer to §8.1, but the shape differs: bridging means the translator maintains both folder layouts; retiring version-in-name means the upgrader does a one-time re-home and version moves into content. **This is the fork that must be settled before §6.3 is implemented.**

---

## 7. Implementation Strategy & Phasing

Sequenced so the riskiest unknowns are proven first and each phase has a gating test.

| Phase | Deliverable | Exit criteria / gate |
|---|---|---|
| **0 — Prove it** *(done)* | [`phase0_repro.py`](./phase0_repro.py) reproducing A/B/C | ✅ Behaviour observed & documented (§3) |
| **1 — Pre-work** | P1–P6 (§5) | Corpus frozen; caches stamped; seam centralized; `_expect_one` reconciles |
| **2 — Translator skeleton + the one real migration** | Registry + folder-naming migration | A 2-folder Drive resolves to one *without* manual surgery |
| **3 — Upgrader + manifest + integrity** | Driver, catalog, re-hash | Solo `0.1.111 → dev` upgrade: manifest written, every asset hash-verified |
| **4 — Peer-skew downcast** | Downcast path keyed on peer `VersionInfo` | Skewed-peer e2e on mock-Drive harness passes |

---

## 8. Decisions Required

The doc's main job is to force these, not bury them.

### 8.1 Folder naming — bridge vs. retire *(blocking §6)*
- **Option A — keep version-scoped folders, bridge them.** Less invasive; translator maintains both layouts. Perpetuates the silo philosophy.
- **Option B — retire version-in-folder-name; version the content instead.** Cleaner long-term; aligns with the translator model; one-time re-home cost. **Recommended.**

### 8.2 Enforce a real `MIN_SUPPORTED` floor in V1? (P2)
Recommend **yes** — it's cheap and V2 needs it.

### 8.3 Sub-package format audit (§4.4) — in V1 scope or tracked separately?
Recommend **audit-in-V1, migrate-in-V2**: we must *know* the surface before claiming V1 is safe, even if we don't migrate it yet.

---

## 9. Proof / Test Strategy

- **Frozen-corpus regression (P5):** captured serialized bytes per supported version; current code must upcast each losslessly. This is what keeps §2.2's "all clean" true as the code moves. Guards against the silent-drop risk of lenient parsing (§2.2).
- **Skewed-peer end-to-end:** on the existing mock-Drive harness (`pair_with_mock_drive_service_connection`), including the flagship `0.1.111 → dev` case and a downcast pair.
- **Asset-integrity invariant test:** `hash(asset) before == hash(asset) after` for every migration — the executable form of §1.3.

---

## 10. Forward-compatibility with V2

V1's primitives are exactly what V2 reuses, *if* we don't cut the wrong corners:

- The **translator registry** generalizes from upcast/downcast-within-minor to the forced-upgrade negotiation — same shape, V2 just adds a "refuse + instruct upgrade" terminal.
- **Version-stamping (P1)** and the **manifest (§6.5)** are version-agnostic; V2 inherits them for free.
- **Enforced `MIN_SUPPORTED` (P2)** *is* the V2 floor.
- **What V1 must NOT do:** hard-code "patch ⇒ always compatible" anywhere outside the compatibility module, or let the upgrader assume a single-minor world. Keep the "is this compatible?" question behind `version_info.compatibility_status_with` so V2 can change the policy in one place.

---

## Appendix A — Evidence index

| Claim | Location |
|---|---|
| Version source of truth | `syft_client/version.py:9-21` |
| Compatibility policy (`SAME`/`PATCH_DIFF`/`INCOMPATIBLE`) | `sync/version/version_info.py:50-66` |
| Folder-name builders (version-stamped) | `gdrive_transport.py:117, 142, 1955` |
| Patch-compat discovery bridge (PR #280) | `gdrive_transport.py:224` (`_filter_patch_compatible`) |
| Multi-folder hard-fail | `gdrive_transport.py:1432-1448` (`_expect_one`) |
| Folder-name break introduced | commit `59b6208c46` (shipped `0.1.112`) |
| `#280` fix | commit `996470dfdf` |
| Wire-message seams | `proposed_filechange.py:106-113`, `file_change_event.py:148-154` |
| Per-model upcast hooks | `proposed_filechange.py:51`, `file_change_event.py:93` |
| Router (de)serialization seam (14 sites) | `sync/connections/connection_router.py` |
| Un-versioned local caches | `persisted_dict.py:96,111`; `peer_store.py:235,239` |
| Schema-version constants (never bumped) | `checkpoint.py:29`, `rolling_state.py:29` |
| Integrity primitive | `file_utils.py:7` (`compute_file_hashes`) |
| Compression wrapper (fixed tar member) | `syftbox_utils.py:62-77` |
| Destructive current upgrade handler | `login_utils.py:75` |

## Appendix B — Glossary
- **Patch cohort:** versions sharing `major.minor` — the V1 translation set.
- **Service-bus manager:** the higher of two patch-skewed peers; owns translation.
- **Upcast / downcast:** translate an old artifact up to current / a current artifact down to an older peer's shape.
- **Stranded asset:** data on disk/Drive that the running client can no longer discover.
