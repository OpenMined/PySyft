# Sharing large private datasets with an enclave: the memory problem and the fix

This explains why `share_private_dataset` on a 5 GB dataset takes half an hour and cannot run on
Google Colab, and what has to change in the sync layer so it can. It assumes you know the notebooks
and roughly how a SyftBox syncs, but not the internals of the sync code.

Everything here was measured on the real classes with a 1 GB incompressible file (model weights do
not compress), then scaled linearly. The script is in the appendix so you can re-run it.

---

## 1. The numbers first

|                                                     | Today                                                    | Needed for the demo                            |
| --------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------- |
| Peak RAM on the model owner's machine, 5 GB dataset | about 20 GB                                              | under 10.5 GB (Colab)                          |
| Peak RAM on the enclave, 5 GB dataset               | about 20 GB                                              | under 7 GB (default `n2d-standard-2` VM, 8 GB) |
| Local CPU work before the upload starts             | about 5.5 min, done twice                                | seconds                                        |
| Behaviour when RAM runs out                         | Laptop: swaps, share takes 30 min. Colab: runtime killed |                                                |

The peak is roughly **4 times the dataset size**, on both ends of the wire. That single ratio is the
whole problem.

---

## 2. The mental model

Think of a private share as putting the dataset in an envelope and mailing it:

```
files on disk
   │  read
   ▼
raw bytes ──► JSON (base64) ──► tar.gz ──► encrypt ──► upload to the enclave's outbox on Drive
```

Every arrow produces a **new copy of the whole payload in memory**, and the old copy is not freed
until the function returns. So at the encrypt step the process is holding the raw bytes, the tar,
the ciphertext, and a scratch buffer the crypto library needs. Four copies. That is the 4x.

The enclave then opens the envelope by running the same chain backwards, so it also holds four
copies while unpacking.

The fix is to make the chain a **stream**: read a small piece, transform it, write it out, forget
it, move to the next piece. Memory then stays at one piece, whatever the dataset size.

---

## 3. Walking through today's code, copy by copy

### 3.1 Read everything into memory (copy 1)

`packages/syft-datasets/src/syft_datasets/dataset_manager.py`, `get_private_dataset_files`:

```python
files = {}
for f in private_dir.rglob("*"):
    ...
    files[path_in_datasite] = f.read_bytes()      # 5 GB of bytes objects
return files
```

The return type is `dict[Path, bytes]`. The whole dataset is now in RAM.

### 3.2 Wrap the bytes in event objects (no new copy, but pinned)

`syft/sync/events/event_cache.py`, `create_events_for_files`:

```python
for path_in_datasite, content in files.items():
    new_hash = get_event_hash_from_content(content)     # one full pass over the bytes
    event = FileChangeEvent(..., content=content, ...)  # same bytes object, no copy
    events.append(event)
return FileChangeEventsMessage(events=events)
```

The `FileChangeEvent` holds a reference to the same bytes, so this does not double memory. But it
means the raw bytes stay alive as long as the message object is alive, which is until the very end
of `share_private_dataset`.

### 3.3 Queue the message, twice

`syft/sync/sync/datasite_owner_syncer.py`:

```python
def queue_event_for_syftbox(self, recipients, file_change_events_message):
    self.syftbox_events_queue.put(file_change_events_message)          # the owner's own event log
    for recipient in recipients:
        self.outbox_queue.put((recipient, file_change_events_message))  # the enclave's outbox

def process_syftbox_events_queue(self):
    while not self.syftbox_events_queue.empty():
        msg = self.syftbox_events_queue.get()
        self.connection_router.owner_write_events_message_to_syftbox(msg)      # serialize+tar+encrypt+upload
    while not self.outbox_queue.empty():
        recipient, msg = self.outbox_queue.get()
        self.connection_router.owner_write_event_messages_to_outbox(recipient, msg)  # ...all of it again
```

`share_private_dataset` calls `queue_event_for_syftbox(recipients=[enclave_email], ...)`. The same
message therefore goes to **two** places: a self-encrypted copy in the owner's own SyftBox event log
on Drive, and the copy for the enclave. Steps 3.4 to 3.7 run once for each. That is why "5.5 minutes
of local work" is really 11, and why the 5 GB upload happens twice.

### 3.4 Base64 the bytes into JSON (copy 2, and it is 33 percent bigger)

`syft/sync/events/file_change_event.py`:

```python
@field_serializer("content", when_used="json")
def serialize_content(self, value):
    return base64.b64encode(value).decode("utf-8")     # 5 GB -> 6.7 GB of text

def as_compressed_data(self) -> bytes:
    return compress_data(self.model_dump_json().encode("utf-8"))
```

Two things happen on that last line. `model_dump_json()` builds a 6.7 GB Python **string**. Then
`.encode("utf-8")` makes a 6.7 GB **bytes** copy of it. For a moment the process holds raw bytes,
the string, and the encoded bytes: 1 + 1.33 + 1.33 = 3.7 times the dataset. This is the first of the
two peaks.

Base64 exists because JSON cannot carry raw bytes. It is the format that forces every later step to
work on a 33 percent larger payload.

### 3.5 Tar and gzip the JSON (copy 3, and the slow step)

`syft/sync/utils/syftbox_utils.py`:

```python
def compress_data(data: bytes) -> bytes:
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="proposed_file_changes.json")
        info.size = len(data)
        tar.addfile(tarinfo=info, fileobj=io.BytesIO(data))
    return tar_bytes.getvalue()                        # another full copy, back to ~5 GB
```

Gzip is single-threaded and runs at about 18 MB/s on this data: **56 seconds per gigabyte**. And it
achieves nothing useful. Model weights are incompressible, so the only thing gzip "compresses" is the
base64 padding, taking 6.7 GB back to 5 GB. We pay five minutes to undo the cost of step 3.4.

### 3.6 Encrypt the whole blob (copy 4, plus a hidden copy 5)

`syft/sync/connections/connection_router.py` and `syft/sync/peers/peer_store.py`:

```python
data = events_message.as_compressed_data()                          # copy 3 (tar)
data = self.peer_store.encrypt_if_needed(recipient_email, data)     # copy 4 (ciphertext)
fname = events_message.message_filepath.as_string()
self.connection_for_outbox().owner_write_raw_bytes_to_outbox(recipient_email, fname, data)
```

```python
def encrypt(self, recipient_email, plaintext: bytes) -> bytes:
    keys = self._ensure_private_keys()
    peer_bundle = self._get_parsed_peer_bundle(recipient_email)
    recipient = syc.EncryptionRecipient(recipient_email, peer_bundle)
    return syc.encrypt_message(self.email, keys, [recipient], plaintext)
```

`syc.encrypt_message` is bytes in, bytes out. There is no streaming variant. Measured, it needs
**three times its input** while it runs: the plaintext, an internal working buffer, and the output.
With the raw bytes still pinned by the message object (3.2), this is the second peak: raw 1 + tar 1 +
internal 1 + ciphertext 1 = **4 times the dataset**.

### 3.7 Upload from memory

`syft/sync/connections/drive/gdrive_transport.py`:

```python
media = MediaIoBaseUpload(io.BytesIO(file_data), mimetype=mime_type, resumable=True)
result = execute_with_retries(
    self.drive_service.files().create(body=file_metadata, media_body=file_payload, fields="id, parents")
)
```

The ciphertext is wrapped in an in-memory stream and sent as one resumable upload. `.execute()`
pushes 100 MB chunks one after another. If the connection drops after 4.9 GB, the retry starts the
whole request again.

### 3.8 The enclave does it all backwards

The watcher side downloads the whole file into memory, decrypts to a full buffer, gunzips to a full
buffer, parses the JSON, base64-decodes every event back to bytes, and only then writes files to
disk. Same four copies, same 4x, on a machine that by default has 8 GB.

### 3.9 Measured, for a 1 GB dataset

| Stage                         | Live objects | Peak RSS so far |
| ----------------------------- | ------------ | --------------- |
| Read into `dict[Path, bytes]` | 1.00 GB      | 1.08 GB         |
| Base64 JSON serialize         | 2.33 GB      | **3.75 GB**     |
| Tar + gzip                    | 3.34 GB      | 3.75 GB         |
| Encrypt                       | 3.00 GB      | **4.03 GB**     |
| Upload                        | 2.00 GB      | 4.03 GB         |

| Stage       | Time per GB |
| ----------- | ----------- |
| Base64 JSON | 3 s         |
| Gzip        | 56 s        |
| Encrypt     | 6 s         |

Multiply by 5 for the 5 GB ShieldGemma share, then by 2 for the duplicate write.

---

## 4. Why small fixes are not enough

It is tempting to fix this with a few local edits. Each helps time, none gets a 5 GB share under the
Colab budget, and here is exactly why.

**"Turn off gzip and free things early."** This removes five minutes per gigabyte. But the encrypt
step still receives a base64 tar of 1.33x and needs three times that: 4x. Same peak.

**"Only chunk the encryption."** Encryption stops being a peak, but step 3.4 still builds the string
and the bytes at once: 3.7x. Same peak, give or take.

**"Chunk the encryption and tidy the serializer."** Serialize straight to bytes with
`pydantic_core.to_json` instead of string-then-encode, free the raw bytes right after, write the tar
to a temp file. Peak falls to about 2.3x, which is 11.7 GB for 5 GB. Still over the 10.5 GB Colab
budget, by a little. And the enclave reader is untouched, so the enclave still peaks at 4x.

The structural reason: **base64 JSON and whole-buffer encryption each force a full copy**, and the
enclave has to be able to undo both. As long as the wire format is "one JSON blob with base64 inside,
encrypted as one message", some step holds the whole payload at least twice. Both have to go.

---

## 5. The fix, part B: stream the payload as a tar of real files

### What changes

The message stops being one JSON document with the files base64'd inside. It becomes a **tar archive
whose members are the files themselves**, plus one small JSON manifest member describing them. Tar
was designed to be written and read as a stream, so nothing needs to be in memory at once.

### Sender

`get_private_dataset_files` returns paths instead of bytes:

```python
# before: dict[Path, bytes]
# after:  dict[Path, Path]   (path_in_datasite -> file on disk)
files[path_in_datasite] = f
```

Events carry the hash and size, not the content:

```python
for path_in_datasite, src in files.items():
    events.append(FileChangeEvent(
        path_in_datasite=path_in_datasite,
        content=None,                         # content travels as a tar member instead
        new_hash=hash_file_streaming(src),    # hash while reading, 1 MB at a time
        size=src.stat().st_size,
        ...
    ))
```

The tar is written to a temp file, member by member, straight from disk:

```python
def write_message_archive(events, files, out_path):
    with tarfile.open(out_path, mode="w") as tar:                 # "w", not "w:gz": no gzip
        manifest = json.dumps({"format": 4, "events": [e.model_dump(mode="json") for e in events]}).encode()
        info = tarfile.TarInfo("manifest.json"); info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
        for e in events:
            src = files[e.path_in_datasite]
            info = tar.gettarinfo(str(src), arcname=str(e.path_in_datasite))
            with open(src, "rb") as fh:
                tar.addfile(info, fh)                             # tarfile copies 16 KB at a time
```

The upload reads from that temp file with an explicit chunk loop, so a failed chunk is retried on
its own rather than restarting the whole upload:

```python
media = MediaFileUpload(str(out_path), mimetype="application/octet-stream",
                        resumable=True, chunksize=256 * 2**20)
request = self.drive_service.files().create(body=file_metadata, media_body=media, fields="id")
response = None
while response is None:
    _, response = next_chunk_with_retries(request)
```

### Receiver (the enclave)

The watcher downloads to a temp file in chunks (the 10 MB `MediaIoBaseDownload` loop already exists
in `download_file`), then extracts members directly to their destination paths:

```python
with tarfile.open(tmp_path, mode="r") as tar:
    manifest = json.load(tar.extractfile("manifest.json"))
    for e in manifest["events"]:
        member = tar.getmember(e["path_in_datasite"])
        dest = datasite_root / e["path_in_datasite"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)          # 1 MB at a time
        verify_hash(dest, e["new_hash"])
```

### Compatibility

The manifest carries `"format": 4`. `FileChangeEventsMessage.from_compressed_data` already routes
old blobs through `load_as_latest`; it gains one branch: if the archive has a `manifest.json` member,
use the new reader, otherwise fall back to the JSON path. Old messages in old outboxes keep working.
A sender only uses the new format when the peer's advertised version supports it, which is the same
negotiation `_private_share_protocol_version` already does for dataset layouts.

### What B buys

Sender and receiver memory for the tar step: constant, about one tar block. Gzip: gone, and with it
56 seconds per gigabyte. Base64: gone, so the payload is 5 GB on the wire instead of 6.7 GB.

What B does **not** fix: step 3.6. `encrypt_message` still needs the whole tar in memory and returns
the whole ciphertext, so with encryption on the peak is still 3x, 15 GB for 5 GB. Which is why part
C exists.

---

## 6. The fix, part C: encrypt in chunks

### The idea

Instead of one envelope around a 5 GB tar, write a short header and then a sequence of small
envelopes, each around a fixed 128 MB slice of the tar. Every slice is a normal `syc.encrypt_message`
call, so no new cryptography is introduced. Only the framing is new.

```
+--------+---------------------+---------------------+-----
| header | len | envelope #1   | len | envelope #2   | ...
+--------+---------------------+---------------------+-----
```

### Sender

```python
CHUNK = 128 * 2**20
MAGIC = b"SYFTCHUNKv1"

def encrypt_file_chunked(self, recipient_email, src_path, dst_path):
    keys = self._ensure_private_keys()
    recipient = syc.EncryptionRecipient(recipient_email, self._get_parsed_peer_bundle(recipient_email))
    n_chunks = -(-src_path.stat().st_size // CHUNK)
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        dst.write(MAGIC + n_chunks.to_bytes(4, "big") + CHUNK.to_bytes(8, "big"))
        for i in range(n_chunks):
            piece = src.read(CHUNK)
            env = syc.encrypt_message(self.email, keys, [recipient], piece,
                                      filename_hint=f"{i}/{n_chunks}")   # binds order into the envelope
            dst.write(len(env).to_bytes(8, "big"))
            dst.write(env)
```

Memory: one 128 MB plaintext piece, plus the crypto library's 3x on that piece. Under half a
gigabyte, whatever the dataset size.

### Receiver

```python
def decrypt_file_chunked(self, sender_email, src_path, dst_path):
    keys = self._ensure_private_keys()
    sender_bundle = self._get_parsed_peer_bundle(sender_email)
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        assert src.read(len(MAGIC)) == MAGIC
        n_chunks = int.from_bytes(src.read(4), "big"); src.read(8)
        for i in range(n_chunks):
            env = src.read(int.from_bytes(src.read(8), "big"))
            parsed = syc.parse_envelope(env)
            syc.verify_envelope_signature(parsed, sender_bundle.identity_key_bytes)
            assert parsed.filename_hint == f"{i}/{n_chunks}"      # reject reordered or dropped chunks
            dst.write(syc.decrypt_message(self.email, keys, sender_bundle, parsed))
```

### Why the order check matters

Each envelope is individually signed and authenticated, so nobody can alter a chunk. But without the
`i/n` tag someone could reorder chunks or drop the last one and the stream would still decrypt. Putting
the index and count inside every envelope, and checking them on the way out, closes that.

### Compatibility

The header's magic bytes distinguish a chunked file from a single envelope, so the receiver tries the
chunked reader first and falls back to `decrypt_message` on the whole file. `_is_syc_envelope` in
`peer_store.py` already does exactly this kind of sniffing for plaintext versus envelope.

---

## 7. Putting B and C together

```
files on disk ──tar members──► temp tar ──128 MB slices──► temp ciphertext ──256 MB chunks──► Drive
     (B)                          (B)              (C)                            (B)
```

|                                   | Today          | B only           | B + C                         |
| --------------------------------- | -------------- | ---------------- | ----------------------------- |
| Sender peak, 5 GB, encryption on  | 20 GB          | 15 GB            | under 1 GB                    |
| Enclave peak, 5 GB, encryption on | 20 GB          | 15 GB            | under 1 GB                    |
| Local CPU before upload           | 5.5 min, twice | seconds          | seconds                       |
| Fits Colab (10.5 GB usable)?      | no             | no               | yes                           |
| Fits default enclave VM (8 GB)?   | no             | no               | yes                           |
| Wire format change                |                | message format 4 | plus chunked envelope framing |

One more line item comes free with B, because the caller changes anyway: `share_private_dataset`
should pass a flag so the message is **not** also written to the owner's own event log. That halves
the work on its own.

---

## 8. Plan

| Step | What                                                                                                                       | Effort      | Files                                                                                                         |
| ---- | -------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------- |
| 0    | Skip the duplicate event-log write for private shares; store instead of gzip                                               | half a day  | `datasite_owner_syncer.py`, `syftbox_utils.py`, `syft_rds/client.py`                                          |
| 1    | B, sender: path-based dataset files, tar-member writer to temp file, `MediaFileUpload` chunk loop                          | 2 days      | `dataset_manager.py`, `event_cache.py`, `file_change_event.py`, `connection_router.py`, `gdrive_transport.py` |
| 2    | B, receiver: manifest-aware reader that extracts members to disk; format negotiation                                       | 1 to 2 days | `file_change_event.py`, `datasite_watcher_syncer.py`, migration entry                                         |
| 3    | C, both sides: chunked envelope writer and reader with order tags and magic header                                         | 3 to 4 days | `peer_store.py`, `connection_router.py`, `gdrive_transport.py`                                                |
| 4    | Tests: round-trip a 1 GB dataset through the in-memory pair and assert peak RSS under 1 GB; old-format messages still load | 1 day       | `syft/tests`, `packages/syft-enclave/tests`                                                                   |

Step 0 ships alone and is safe today. Steps 1 and 2 are one PR because a sender without a reader is
useless. Step 3 is a second PR. Total is roughly two weeks.

---

## Appendix: the measurement script

Run from the repo root with the project venv. It uses the real `FileChangeEvent`,
`FileChangeEventsMessage`, `compress_data`, and `syft_crypto_python`, so the numbers track the code.

```python
import os, io, time, resource, gc
from pathlib import Path
from uuid import uuid4
import psutil
from syft.sync.events.file_change_event import FileChangeEvent, FileChangeEventsMessage
from syft.sync.utils.syftbox_utils import create_event_timestamp, compress_data
import syft_crypto_python as syc

proc, GB = psutil.Process(), 2**30
rss  = lambda: proc.memory_info().rss / GB
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB      # bytes on macOS
row  = lambda label, live: print(f"{label:40s} live={live/GB:5.2f}G rss={rss():5.2f}G peak={peak():5.2f}G")

N = 1 * GB
content = os.urandom(N)                                                     # incompressible, like weights
row("1 read", N)
ev = FileChangeEvent(id=uuid4(), path_in_datasite=Path("private/x/model.safetensors"), content=content,
                     old_hash=None, new_hash="h", submitted_timestamp=create_event_timestamp(),
                     timestamp=create_event_timestamp(), datasite_email="a@example.com", is_deleted=False)
msg = FileChangeEventsMessage(events=[ev])
t = time.time(); js = msg.model_dump_json().encode(); row(f"3a base64 json ({time.time()-t:.0f}s)", N + len(js))
t = time.time(); gz = compress_data(js);            row(f"3b tar.gz ({time.time()-t:.0f}s)", N + len(js) + len(gz))
del js; gc.collect()
keys = syc.SyftRecoveryKey.generate().derive_keys()
doc = keys.to_public_bundle().to_did_document("did:syft:a@example.com"); doc["identity"] = "a@example.com"
rec = syc.EncryptionRecipient("a@example.com", syc.SyftPublicKeyBundle.from_did_document(doc))
t = time.time(); env = syc.encrypt_message("a@example.com", keys, [rec], gz); row(f"4 encrypt ({time.time()-t:.0f}s)", N + len(gz) + len(env))
print(f"\npeak for {N/GB:.0f} GB: {peak():.2f} GB  ->  x5 for 5 GB: ~{peak()*5:.0f} GB")
```

---

## Status: implemented as "B2, share the private collection"

Private datasets no longer travel through event messages. `share_private_dataset` now:

1. streams the dataset's files from disk (no `read_bytes`),
2. seals each file once for the owner **and** the recipient with the streaming cipher suite of
   `syft-crypto-python >= 0.1.2b4` (`encrypt_file`, one envelope, one key wrapping per recipient),
3. uploads them as a **private collection** named by content hash plus recipient set, so a repeat
   share for the same audience uploads nothing and a new audience gets its own collection,
4. shares the collection folder with the recipient by Drive permission.

The enclave's watcher pulls private collections that were shared with it
(`CollectionSyncSpec.pull_when_shared`), streaming each file to disk and decrypting it into
`<owner>/private/syft_datasets/[v<n>/]<tag>/`, where jobs already resolve it. The owner's own
cold-start restore goes through the same streamed, self-decrypting path, which also fixes the
backup upload that used to buffer whole files.

Measured on the peer_store path: RSS growth is flat at about 19 MiB for 48, 96, and 192 MiB
files (the parallel segment buffers), independent of file size. The design comparison is in
`private-dataset-transport-options.md`.

What did not change: the notebooks, the job code, the event message format, the migration
registry. The gzip/own-log trims proposed earlier were dropped as moot.
