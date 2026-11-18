# Sync flow

Every change travels as a message through folders on Google Drive. A DS writes proposed changes to
the DO's inbox. The DO writes the accepted changes, and its own changes, to one outbox per DS.

## DS: proposing a change

A DS changes a file in a DO's datasite by proposing the change. `DatasiteWatcherSyncer.on_file_change()`
queues the path, and `DatasiteWatcherSyncer.process_file_changes_queue()` sends one
`ProposedFileChangesMessage` to each DO. Each change carries the hash of the file that the DS last
saw. This happens straight away, not in `sync()`. `SyftboxManager.push_job_files()` uses it to
submit a job.

## One sync pass

`SyftboxManager.sync()` runs one pass. It takes a file lock (`<syftbox>/.sync.lock`), so two
processes do not sync the same syftbox at once, and loads the peers. Then it runs the DO part when
the manager has the DO role, and the DS part when it has the DS role.

### DO part

`DatasiteOwnerSyncer.sync()` handles the approved peers whose version is compatible. It runs these
steps in order:

1. **Restore state.** On the first pass, the DO restores its own state from Drive (checkpoints,
   rolling state and the newer events) and its collections.
2. **Record local changes.** The DO hashes its local files and compares them with the cached hashes.
   Each difference becomes a `FileChangeEvent`. This runs before the proposals, so that a proposal
   based on an older version of a file the DO changed is refused as outdated.
3. **Pull proposals.** For each peer, the DO reads the proposals in its inbox. It keeps only changes
   to syncable paths that the sender may write. When the hash in a change differs from the DO's
   current hash, the whole proposal is refused as outdated. Every proposal is archived afterwards,
   applied or not, so a bad one does not hold back the next.
4. **Save the owner state.** The DO writes the new events to its event log on Drive, and updates its
   rolling state. The DO reads these back only to restore its datasite, for example on a new
   machine.
5. **Deliver.** `DeliveryTracker` keeps a view of the files each peer holds. For each peer, the DO
   plans to send the files the peer may read but lacks, and to delete the files it may no longer
   have. The plan goes out as `FileChangeEventsMessage`s in that peer's outbox. A failed send is
   tried again, up to 3 times in one pass. A peer that is not ready gets nothing in this pass.

After the pass, the DO compacts an outbox that holds 20 or more message files, and creates a
checkpoint once 50 or more events have built up.

### DS part

`DatasiteWatcherSyncer.sync_down()` reads each DO's outbox, starting after the last message it
received, plus any message that landed late. It applies only events under that DO's own folder,
`<syftbox>/<do_email>/`, and drops the rest. Then it downloads the shared collections, such as
datasets, in parallel.

## Events and messages

The DO stores all its state as `FileChangeEvent` objects, in `FileChangeEventsMessage` batches.
Each event contains:

- `id`: UUID for the change
- `path_in_datasite`: file path relative to the datasite
- `datasite_email`: the datasite the file belongs to
- `content`: file content (text or binary, base64-encoded for transport)
- `old_hash`/`new_hash`: for change detection and conflict resolution. `new_hash` is None for a
  deletion.
- `is_deleted`: flag for deletion events
- `submitted_timestamp`/`timestamp`: when the change was submitted, and when the event was created

An events message is compressed as a `.tar.gz` file named
`syfteventsmessagev3_{timestamp}_{uuid}.tar.gz`. A proposal file name starts with `msgv2`. Before
a message is sent to a peer, it is converted to the protocol version the peer speaks, and encrypted
when the pair uses encryption.
