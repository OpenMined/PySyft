# Moving large private datasets to the enclave: three designs

Companion to `large-private-dataset-sharing.md`, which explains _why_ `share_private_dataset`
peaks at ~4× the dataset size. This document compares the three ways to fix the transport, at
file and function level, and ends with a recommendation. The streaming cipher it relies on
(`encrypt_file` / `decrypt_file` in `syft-crypto-python >= 0.1.2b4`) is already released and
pinned.

---

## 1. The two channels that exist today

The sync engine (`syft/sync`) moves data between SyftBoxes over Drive in two different ways.

**Event messages** (`syft/sync/events`, `syft/sync/sync/datasite_owner_syncer.py`,
`datasite_watcher_syncer.py`). One JSON document per change, dropped into a per-peer outbox
folder `syft_datasite#<ver>#<owner>#outbox#<peer>`. The receiver downloads it whole, decrypts
it whole, parses it whole, stores a copy in `<syftbox>-event-messages/`, then writes each
event's `content` to `<owner>/<path_in_datasite>`. Everything travels _inside_ the message, so
binary content is base64'd. This is the control plane: permissions, metadata, job scripts,
job results. `share_private_dataset` uses it, which is the problem.

**Collections** (`syft/sync/sync/collection_spec.py`, `datasite_watcher_cache.py`, the
`*_collection*` methods in `gdrive_transport.py`). A named, content-hashed **folder** on Drive,
`<prefix>_<tag>_<hash>`, with one Drive file per dataset file. The owner uploads files one at a
time (optionally encrypted per recipient), shares the folder by Drive permission, and a watcher
lists folders it can see, skips any whose hash it already holds, downloads files in parallel
with a thread pool, and writes each to `<owner>/<subpath>/<tag>/<file>`. This is the data plane.
Mock datasets travel this way. Private datasets are _also_ uploaded this way, as an owner-only,
self-encrypted backup (`upload_private=True`, `PRIVATE_DATASET_SPEC`, `owner_only=True`) that
watchers are told to skip.

Both channels share one transport implementation, `gdrive_transport.py`, driven either by the
real Drive API or by `mock_drive_service.py` in the in-memory notebooks and tests. So every
transport change below is written once and exercised by the in-memory quad.

The backup path has the same class of problem: `_upload_private_dataset_to_collection` does
`f.read_bytes()` per file and `encrypt_if_needed` on the whole file, so the 4.9 GB ShieldGemma
shard costs ~15 GB on `create_dataset(upload_private=True)` today.

---

## 2. Option A: tar members inside the event message

### Shape

The message stays one object in the outbox, but it becomes a real tar: a `manifest.json` member
holding the events (with `content: null`), then one member per file streamed from disk.

```
syfteventsmessagev4_<ts>_<uuid>.tar          (encrypted with encrypt_file, streaming suite)
  manifest.json                              3 KB
  private/syft_datasets/v1/shieldgemma_model/model-00001-of-00002.safetensors   4.9 GB
  private/syft_datasets/v1/shieldgemma_model/model-00002-of-00002.safetensors   240 MB
  ...
```

### Sender

```python
# syft_rds/client.py  share_private_dataset
paths = self.dataset_manager.get_private_dataset_paths(tag, protocol_version)   # dict[Path, Path]
events = event_cache.create_events_for_paths(paths)                              # hashes streaming, content=None
archive = write_message_archive(events, paths)                                   # temp tar on disk
self.sync_engine.datasite_owner_syncer.queue_archive_for_outbox(enclave_email, archive)
```

```python
# syft/sync/utils/syftbox_utils.py
def write_message_archive(events, paths, out_path):
    with tarfile.open(out_path, "w") as tar:
        manifest = json.dumps({"format": 4, "events": [e.model_dump(mode="json") for e in events]}).encode()
        info = tarfile.TarInfo("manifest.json"); info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
        for e in events:
            src = paths[e.path_in_datasite]
            with open(src, "rb") as fh:
                tar.addfile(tar.gettarinfo(str(src), arcname=str(e.path_in_datasite)), fh)
```

```python
# syft/sync/connections/connection_router.py
def owner_write_archive_to_outbox(self, recipient, archive_path):
    enc = archive_path.with_suffix(".syc")
    self.peer_store.encrypt_file(recipient, archive_path, enc)          # new, streaming
    self.connection_for_outbox().owner_upload_file_to_outbox(recipient, fname, enc)   # MediaFileUpload
```

### Receiver

```python
# syft/sync/sync/datasite_watcher_syncer.py
def download_events_message_with_new_connection(self, file_id, peer_email):
    tmp = connection.download_file_to_path(file_id)                     # 10 MB chunks to disk, not BytesIO
    plain = self.peer_store.decrypt_file(peer_email, tmp)               # new, streaming
    return FileChangeEventsMessage.from_archive(plain)                   # manifest only; members stay on disk

# syft/sync/sync/caches/datasite_watcher_cache.py  apply_event_message
if message.archive_path:
    with tarfile.open(message.archive_path) as tar:
        for e in message.events:
            with tar.extractfile(str(e.path_in_datasite)) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            verify_hash(dest, e.new_hash)
    self.events_connection.write_file(name, message.without_archive())  # cache the manifest, not 5 GB
```

### Files touched

| File                                              | Change                                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `packages/syft-datasets/.../dataset_manager.py`   | `get_private_dataset_paths`                                                            |
| `syft/sync/events/event_cache.py`                 | `create_events_for_paths` with streaming hash                                          |
| `syft/sync/events/file_change_event.py`           | message format 4; `from_archive` / `without_archive`; `archive_path` transient field   |
| `syft/sync/utils/syftbox_utils.py`                | `write_message_archive`, `read_manifest`                                               |
| `syft/sync/connections/base_connection.py`        | `owner_upload_file_to_outbox`, `download_file_to_path`                                 |
| `syft/sync/connections/drive/gdrive_transport.py` | implement both with `MediaFileUpload` chunk loop and file-backed `MediaIoBaseDownload` |
| `syft/sync/connections/connection_router.py`      | `owner_write_archive_to_outbox`, file-based decrypt on read                            |
| `syft/sync/peers/peer_store.py`                   | `encrypt_file`, `decrypt_file`, `verify_file` wrappers                                 |
| `syft/sync/sync/datasite_owner_syncer.py`         | `queue_archive_for_outbox`; **outbox compaction must skip archives**                   |
| `syft/sync/sync/datasite_watcher_syncer.py`       | download to path, decrypt file, parse manifest                                         |
| `syft/sync/sync/caches/datasite_watcher_cache.py` | extract members to disk; cache manifest only                                           |
| `syft/sync/version/peer_manager.py`               | advertise format 4 so old peers never receive it                                       |
| `syft/migrations/`                                | registry entry for the new message version                                             |
| `packages/syft-rds/src/syft_rds/client.py`        | `share_private_dataset` builds from paths                                              |

### Consequences

- The message pipeline itself becomes partly file-based. Every reader of messages (download,
  decrypt, parse, cache, compaction) gains a second layout to handle.
- One artifact per share, atomic by construction.
- A failed upload restarts the archive; an unchanged re-share re-uploads everything.
- The private backup path is untouched and remains a ~15 GB operation for the shard.

---

## 3. Option B1: by reference, loose blobs in the outbox

### Shape

Each file is encrypted to a temp file and uploaded as its own Drive file into the outbox. The
event message is unchanged in format except for one optional pointer field, and stays ~3 KB.

```
syft_datasite#…#outbox#enclave/
  blob_<sha256-of-model-00001>.bin     4.9 GB   ciphertext
  blob_<sha256-of-model-00002>.bin     240 MB   ciphertext
  ...
  syfteventsmessagev3_<ts>_<uuid>.tar.gz   3 KB   events with blob_ref, content=null
```

### Sender

```python
# syft_rds/client.py  share_private_dataset
paths = self.dataset_manager.get_private_dataset_paths(tag, protocol_version)
events = []
for path_in_datasite, src in paths.items():
    digest = hash_file_streaming(src)
    enc = tmp / f"blob_{digest}.bin"
    self.peer_store.encrypt_file(enclave_email, src, enc)
    file_id = router.owner_upload_blob_to_outbox(enclave_email, enc.name, enc)      # MediaFileUpload
    events.append(FileChangeEvent(path_in_datasite=path_in_datasite, content=None, new_hash=digest,
                                  blob_ref=BlobRef(drive_file_id=file_id, size=src.stat().st_size), ...))
router.owner_write_event_messages_to_outbox(enclave_email, FileChangeEventsMessage(events=events))  # existing path
```

Blobs are uploaded before the message, so a pointer always resolves.

### Receiver

```python
# datasite_watcher_cache.py  apply_event_message
for event in message.events:
    if event.blob_ref is not None:
        enc = router.watcher_download_blob_to_file(event.blob_ref.drive_file_id)   # existing 10 MB loop, to disk
        self.peer_store.decrypt_file(peer_email, enc, dest)                         # streaming
        verify_hash(dest, event.new_hash)
    else:
        self.file_connection.write_file(str(event.path_in_syftbox), event.content)  # today's path
```

### Files touched

| File                                              | Change                                                                     |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `packages/syft-datasets/.../dataset_manager.py`   | `get_private_dataset_paths`                                                |
| `syft/sync/events/file_change_event.py`           | `BlobRef`; `FileChangeEventV2` with optional `blob_ref`; message version 2 |
| `syft/migrations/`                                | v1→v2 entry                                                                |
| `syft/sync/events/event_cache.py`                 | `create_events_for_paths`                                                  |
| `syft/sync/connections/base_connection.py`        | `owner_upload_blob_to_outbox`, `watcher_download_blob_to_file`             |
| `syft/sync/connections/drive/gdrive_transport.py` | implement both                                                             |
| `syft/sync/connections/connection_router.py`      | wrappers with file encrypt/decrypt                                         |
| `syft/sync/peers/peer_store.py`                   | `encrypt_file`, `decrypt_file`, `verify_file`                              |
| `syft/sync/sync/caches/datasite_watcher_cache.py` | blob branch in `apply_event_message`; cache without content                |
| `syft/sync/version/peer_manager.py`               | advertise blob-ref support                                                 |
| `packages/syft-rds/src/syft_rds/client.py`        | orchestration above                                                        |

### Consequences

- The message pipeline is untouched; a side channel is added beside it.
- Per-file retry; re-share can skip blobs whose hash already exists in the outbox.
- It re-implements, inside the outbox, what collections already provide: per-file upload,
  per-recipient encryption, hash naming, parallel download. Two mechanisms for the same job.
- The private backup path is untouched.

---

## 4. Option B2: share the private collection

### Shape

Nothing new on Drive. The private collection that `upload_private` already creates becomes the
share: its files are encrypted for the owner **and** the enclave (one ciphertext, two key
wrappings, the envelope format supports this natively), and the folder is shared with the
enclave by Drive permission, exactly as mock collections are shared with data scientists.

```
syft_privatecollection[_v1]_shieldgemma_model_<hash>/     folder shared with enclave@…
  model-00001-of-00002.safetensors      4.9 GB  envelope, recipients = [owner, enclave]
  model-00002-of-00002.safetensors      240 MB
  config.json … shield_inference.py     KB
  private_metadata.yaml
```

The enclave's watcher pulls it into `<owner>/private/syft_datasets/[v1/]<tag>/`, which is where
`sy.resolve_dataset_files_path` already looks inside a job. No event message carries data; an
optional tiny event can announce "dataset X shared" for UX.

### Sender

```python
# syft_rds/client.py
def _upload_private_dataset_to_collection(self, dataset, recipients: list[str]) -> str | None:
    paths = {f.name: f for f in dataset.private_dir.iterdir() if f.is_file()}
    if not paths:
        return None
    content_hash = CollectionFolder.compute_hash_from_paths(paths)          # streaming sha256
    wire_prefix = PRIVATE_DATASET_SPEC.wire_prefix(dataset_variant(dataset.protocol_version))
    folder_id = self.sync_engine.create_collection_folder(wire_prefix, tag=dataset.name, content_hash=content_hash)
    self.sync_engine.upload_collection_files(wire_prefix, dataset.name, content_hash, paths,
                                             recipients=[self.email, *recipients])
    return folder_id

def share_private_dataset(self, tag: str, enclave_email: str):
    protocol_version = self._private_share_protocol_version(tag, enclave_email)
    dataset = self.dataset_manager.get(tag, protocol_version=protocol_version)
    self._upload_private_dataset_to_collection(dataset, recipients=[enclave_email])   # no-op if hash+recipients unchanged
    self.sync_engine.share_collection(wire_prefix, tag, content_hash, [enclave_email])
```

```python
# syft/sync/connections/connection_router.py
def owner_upload_collection_files(self, prefix, tag, content_hash, paths: dict[str, Path], recipients=None):
    to_upload = {}
    for name, src in paths.items():
        if recipients and self.peer_store:
            enc = self._tmp / name
            self.peer_store.encrypt_file_for(recipients, src, enc)   # one envelope, N wrappings
            to_upload[name] = enc
        else:
            to_upload[name] = src
    self.connection_for_send_message().owner_upload_collection_files(prefix, tag, content_hash, to_upload)
```

```python
# syft/sync/connections/drive/gdrive_transport.py
def owner_upload_collection_files(self, prefix, tag, content_hash, paths: dict[str, Path]):
    folder_id = self._get_collection_folder_id(prefix, tag, content_hash)
    for name, path in paths.items():
        media = MediaFileUpload(str(path), mimetype="application/octet-stream", resumable=True, chunksize=256 << 20)
        request = self.drive_service.files().create(body={"name": name, "parents": [folder_id]}, media_body=media, fields="id")
        response = None
        while response is None:
            _, response = next_chunk_with_retries(request)
```

### Receiver

```python
# syft/sync/sync/collection_spec.py
class CollectionSyncSpec(BaseModel):
    ...
    owner_only: bool = False        # unchanged meaning for the owner's own backup
    pull_when_shared: bool = False  # NEW: a peer's watcher pulls this collection if the owner
                                    # explicitly shared the folder with it (Drive permission)
PRIVATE_DATASET_SPEC = CollectionSyncSpec.private(..., pull_when_shared=True)
```

```python
# datasite_watcher_cache.py  sync_down_collections_parallel
for spec in self.collection_specs:
    if spec.owner_only and not spec.pull_when_shared:
        continue                                   # was: skip every owner-only spec
    collections = router.watcher_list_collections(spec.prefix)   # Drive search only returns folders shared with me
    ...
    for (collection, metadata) in all_downloads:
        enc = router.watcher_download_collection_file_to_path(metadata["file_id"])   # existing 10 MB loop, to disk
        dest = self.syftbox_folder / owner_email / subpath / tag / metadata["file_name"]
        self.peer_store.decrypt_file(owner_email, enc, dest)                          # streaming
```

Discovery is already the right shape: `watcher_list_collections` queries Drive for folders with
the prefix that are _not_ owned by me, and Drive only returns folders that were shared with me.
The owner's own backup is never shared, so no other peer can see it; the enclave sees exactly
the collections shared with it.

### Files touched

| File                                              | Change                                                                                                                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/syft-rds/src/syft_rds/client.py`        | `_upload_private_dataset_to_collection(dataset, recipients)` from paths; `share_private_dataset` becomes upload-if-needed + share folder; `_collect_*_files` return paths |
| `packages/syft-rds/src/syft_rds/config.py`        | `PRIVATE_DATASET_SPEC` gets `pull_when_shared=True`                                                                                                                       |
| `syft/sync/sync/collection_spec.py`               | `pull_when_shared` flag                                                                                                                                                   |
| `syft/sync/syftbox_manager.py`                    | `upload_collection_files(..., paths, recipients)` pass-through                                                                                                            |
| `syft/sync/connections/base_connection.py`        | `owner_upload_collection_files` takes paths; `watcher_download_collection_file_to_path`                                                                                   |
| `syft/sync/connections/drive/gdrive_transport.py` | implement both; `CollectionFolder.compute_hash_from_paths`                                                                                                                |
| `syft/sync/file_utils.py`                         | `compute_file_hashes_from_paths` (streaming)                                                                                                                              |
| `syft/sync/connections/connection_router.py`      | per-file `encrypt_file_for(recipients)` before upload; `decrypt_file` after download                                                                                      |
| `syft/sync/peers/peer_store.py`                   | `encrypt_file_for(recipients, src, dst)`, `decrypt_file(sender, src, dst)`, `verify_file`                                                                                 |
| `syft/sync/sync/caches/datasite_watcher_cache.py` | pull shared private collections; write via decrypt-to-path; keep hash cache                                                                                               |
| `syft/sync/sync/datasite_owner_syncer.py`         | `pull_initial_state` restores the backup through the same path-based download                                                                                             |

Not touched: `file_change_event.py`, the migration registry, the events cache, compaction,
`datasite_watcher_syncer.py`'s message path, `peer_manager.py` version negotiation (the folder
is simply invisible to peers that were not shared on it).

### Consequences

- One mechanism for all dataset movement, mock and private alike. The events channel is left to
  what it is good at: small control-plane messages.
- Content-hash naming gives dedup for free: re-sharing unchanged weights uploads nothing, and
  the watcher's `collection_hashes` skip already exists.
- The private backup is fixed by the same code, and the backup and the share are the _same_
  upload when the recipients are known at creation time.
- Adding a recipient later (a second enclave) re-uploads under a new hash. A follow-up in
  syft-crypto-core (header/body split plus a "rewrap for one more recipient" operation) would
  make that a 4 KB header upload. Nothing in B2's layout has to change for it.
- Deleting the dataset already deletes the private collection (`delete_dataset` does it today).

---

## 5. Side by side

|                                   | A. Tar members                              | B1. Loose blobs           | B2. Collections                                 |
| --------------------------------- | ------------------------------------------- | ------------------------- | ----------------------------------------------- |
| Peak memory, both ends            | one segment                                 | one segment               | one segment                                     |
| Sender disk passes at 5 GB        | 2 (tar, encrypt)                            | 1 (encrypt)               | 1 (encrypt)                                     |
| Receiver disk passes              | 2 (decrypt, extract)                        | 1                         | 1                                               |
| Upload parallelism                | one stream                                  | per file                  | per file                                        |
| Download parallelism              | one stream                                  | per file                  | per file, exists today                          |
| Message format change             | yes, new layout + migration                 | one field + migration     | none                                            |
| Message pipeline touched          | download, decrypt, parse, cache, compaction | none                      | none                                            |
| Version negotiation needed        | yes                                         | yes                       | no (invisible to unshared peers)                |
| Fixes `upload_private` backup too | no                                          | no                        | yes                                             |
| Dedup on re-share                 | no                                          | possible                  | built in                                        |
| Atomicity                         | one archive                                 | blobs first, message last | folder named by hash, shared only once complete |
| New concepts introduced           | archive messages                            | blob pointers             | one spec flag                                   |
| Rough effort                      | ~1 week                                     | ~1 week                   | ~1 week                                         |

---

## 6. Recommendation

> **Implemented:** B2 landed as described in section 4, with dedup by name (recipient set folded
> into the collection hash) and without the header rewrap; adding a peer later re-encrypts into a
> new collection. See the status section of `large-private-dataset-sharing.md`.

**B2.** The reasoning, as I would defend it in a design review:

1. **It matches the architecture the codebase already has.** The sync engine was refactored in
   July into a control plane (events) and a data plane (collections). Private sharing is the one
   bulk-data flow still on the control plane, for historical reasons only (it predates generic
   collections by four months). Moving it is completing that refactor, not adding to it.
2. **It removes a mechanism instead of adding one.** Today private data is uploaded twice through
   two channels. After B2 there is one upload, readable by everyone entitled to it. A and B1 both
   leave the duplicate in place and add a third way to move bytes.
3. **It does not touch the message format.** No migration entry, no version negotiation, no
   second reader in the events pipeline, no compaction special case. The blast radius is the
   collections code and `share_private_dataset`.
4. **The sharing primitive is Drive permissions**, which is already how every mock dataset is
   shared, already audited, and already understood by the people who operate this.
5. **It fixes the backup bug we found on the way**, which A and B1 would leave for another day.
6. **Its follow-ups are cheap and local.** Multi-enclave sharing without re-upload is a header
   rewrap in the crypto library, not a transport change.

A is the right choice only if a single self-contained archive per share is a hard requirement,
for example for auditing or air-gapped transfer. B1 is never the best choice: it is B2 with the
collection machinery re-implemented in the outbox.

### Suggested order of work for B2

1. `peer_store`: `encrypt_file_for`, `decrypt_file`, `verify_file`. Half a day, tests with two
   PeerStores and a file larger than the test's memory budget.
2. Transport: path-based `owner_upload_collection_files` and `watcher_download_collection_file_to_path`
   in `gdrive_transport.py`, exercised through `mock_drive_service.py`. One day.
3. `file_utils.compute_file_hashes_from_paths`, `CollectionFolder.compute_hash_from_paths`. Hours.
4. `syft_rds/client.py`: paths instead of bytes in the three `_collect_*` / `_upload_*` helpers;
   `share_private_dataset` rewritten to upload-if-needed plus share. One day.
5. `collection_spec.pull_when_shared` and the watcher change, plus `pull_initial_state`. One day.
6. In-memory quad test: create dataset with a 100 MB private file, share with the enclave, assert
   the enclave resolves it, peak RSS bounded, second share uploads nothing. Half a day.
7. Remove the events-based body of `share_private_dataset` once the notebooks pass. Keep the
   quick trims (gzip level, own-log skip) as an independent PR; they still help small messages.
