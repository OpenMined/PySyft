# Connections — Google Drive Transport

Syft-client uses Google Drive as its file-based transport layer. All peer-to-peer communication happens through shared GDrive folders.

## Folder Structure & Permissions

**Per-peer folders** — one inbox and one outbox per peer connection:

```
/SyftBox/
├── syft_datasite#{peer}#inbox#{me}/                   # DS creates, shares write with DO
├── syft_datasite#{peer}#outbox#{me}/                  # DS creates, shares write with DO
```

**Aggregated state** — combined data from all peers, stored on the DO's own drive:

```
/SyftBox/
├── {my_email}/                                        # Event log (all peers)
├── {my_email}-checkpoints/                            # Checkpoints (all peers)
├── {my_email}-rolling-state/                          # Rolling state (all peers)
```

**Dataset folders:**

```
/SyftBox/
├── syft_datasetcollection_{tag}_{hash}/               # Mock data — shared as reader with DS
├── syft_privatecollection_{tag}_{hash}/               # Private data — owner-only, never shared
```

**Encryption bundles** — public keys only, no secrets:

```
/SyftBox/
└── syft_encryption_bundles#{email}/                   # Public keys — shared as reader with peers
    └── encryption_bundle_{owner}_for_{peer}.json
```

| Folder                                        | Scope       | Owner | Shared with | Access |
| --------------------------------------------- | ----------- | ----- | ----------- | ------ |
| `inbox`                                       | per peer    | DS    | DO          | writer |
| `outbox`                                      | per peer    | DS    | DO          | writer |
| `event log` / `checkpoints` / `rolling-state` | all peers   | DO    | nobody      | —      |
| `datasetcollection`                           | per dataset | DO    | DS users    | reader |
| `privatecollection`                           | per dataset | DO    | nobody      | —      |
| `encryption_bundles`                          | per user    | user  | peers       | reader |

---

## Peer Requests

DS creates inbox/outbox folders on their own drive and grants the DO write access. The DO discovers these folders by searching for `syft_datasite#{my_email}#` folders they don't own.

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant DS_Drive as DS's GDrive
    participant DO_Drive as DO's GDrive
    participant DO as Data Owner

    DS->>DS_Drive: Create inbox & outbox folders
    DS->>DS_Drive: Grant DO write access
    DS->>DS_Drive: Record peer as REQUESTED_BY_ME

    DO->>DS_Drive: Search for shared folders matching my email
    DO-->>DO: Peer appears as pending request

    alt Approve
        DO->>DO_Drive: Create reciprocal inbox & outbox
        DO->>DO_Drive: Grant DS write access
        DO->>DO_Drive: Record peer as ACCEPTED
    else Reject
        DO->>DO_Drive: Record peer as REJECTED
    end
```

---

## Receiving Files (DS → DO)

Each peer has its own inbox folder. The DO downloads proposed changes from a single peer's inbox, validates permissions, applies to the local cache, and uploads the resulting events to the aggregated SyftBox event log (which contains state from all peers).

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant Inbox as Inbox (DS's Drive)
    participant DO as Data Owner
    participant Cache as DO Local Cache
    participant SyftBox as SyftBox (DO's GDrive)

    DS->>Inbox: Upload msgv2_{ts}_{uid}.tar.gz
    DO->>Inbox: List & download oldest message
    DO->>DO: Decrypt & deserialize
    DO->>DO: Check write permissions per file
    DO->>Cache: Apply accepted changes
    DO->>SyftBox: Upload events (encrypted for self)
```

---

## Sending Changes (DO → DS)

The aggregated SyftBox event log contains state from all peers. The DO pushes events from it to each peer's individual outbox folder.

```mermaid
sequenceDiagram
    participant DO as Data Owner
    participant SyftBox as /SyftBox/{email}/
    participant Outbox as Outbox (DS's Drive)
    participant DS as Data Scientist
    participant Cache as DS Local Cache

    DO->>SyftBox: Upload events (encrypted for self)
    DO->>Outbox: Upload events (encrypted for DS)
    DS->>Outbox: List new event files (since last sync)
    DS->>Outbox: Download in parallel
    DS->>DS: Decrypt & deserialize
    DS->>Cache: Apply events to local cache
```

---

## Uploading Datasets

Mock data is shared with DS users as readers. Private data stays owner-only.

```mermaid
sequenceDiagram
    participant DO as Data Owner
    participant Mock as Mock Folder (GDrive)
    participant Private as Private Folder (GDrive)
    participant DS as Data Scientist

    DO->>Mock: Create datasetcollection folder
    DO->>Mock: Upload mock files (encrypted per recipient)
    DO->>Mock: Share folder as reader with DS users

    DO->>Private: Create privatecollection folder
    DO->>Private: Upload private files (encrypted for self)
    Note right of Private: Never shared — owner-only

    DS->>Mock: Search for datasetcollection folders shared with me
    DS->>Mock: Download & decrypt mock files
```

---

## Checkpoints

The DO's state is built from **file change events** — each event records a file path, its content, and hashes. As events accumulate, we need to persist them so we can restore state without replaying everything from scratch.

**The problem:** Google Drive has strict rate limits (~50 files per listing). Storing each event as a separate GDrive file quickly hits that limit. But compacting everything into a single large file and re-uploading it on every change makes individual API calls slow.

**The solution:** Three tiers that balance upload size vs. file count:

```mermaid
sequenceDiagram
    participant Local as DO Local
    participant GDrive as GDrive (rolling-state / checkpoints)

    Note over Local: New file change event arrives

    Local->>Local: Append event to in-memory rolling state
    Local->>GDrive: Upload rolling state (single file, updated in-place)

    Note over Local: Rolling state reaches 50 events

    Local->>Local: Promote rolling state → incremental checkpoint
    Local->>GDrive: Upload incremental checkpoint
    Local->>Local: Clear rolling state

    Note over Local: After 4 incremental checkpoints

    Local->>Local: Merge all incrementals → full checkpoint (snapshot of all file states)
    Local->>GDrive: Upload full checkpoint
    Local->>GDrive: Delete old full checkpoint & incrementals
```

| Tier              | What it stores                      | Trigger                         | GDrive file                                    | Retained        |
| ----------------- | ----------------------------------- | ------------------------------- | ---------------------------------------------- | --------------- |
| **Rolling state** | Recent events since last checkpoint | Every event                     | `rolling_state_{ts}.tar.gz` (updated in-place) | 1               |
| **Incremental**   | Batch of 50 events                  | Rolling state reaches 50 events | `incremental_checkpoint_{seq}_{ts}.tar.gz`     | Up to 4         |
| **Full**          | Snapshot of all file states         | 4 incrementals accumulated      | `checkpoint_{ts}.tar.gz`                       | 1 (old deleted) |

**File structures** (all stored as compressed `.tar.gz`):

```jsonc
// Rolling State & Incremental Checkpoint — store events
{
  "email": "do@org.com",
  "timestamp": 1234567890.0,
  "events": [
    {
      "path_in_datasite": "public/data.csv",
      "content": "...",
      "old_hash": "abc...",
      "new_hash": "def...",
      "is_deleted": false,
      "timestamp": 1234567889.0
    }
  ]
}

// Full Checkpoint — stores file snapshots (no event history)
{
  "email": "do@org.com",
  "timestamp": 1234567890.0,
  "last_event_timestamp": 1234567889.0,
  "files": [
    {
      "path": "public/data.csv",
      "hash": "def...",
      "content": "..."
    }
  ]
}
```

**Restore sequence** on initial sync:

1. Download latest **full checkpoint** from GDrive → apply all file states to local cache
2. Download **incremental checkpoints** (sorted by sequence) → replay events on top
3. Download **rolling state** → replay remaining events
4. Download any **event log files** newer than last known timestamp

All checkpoint data is encrypted for self before upload.
