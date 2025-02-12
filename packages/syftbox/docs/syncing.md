# Syncing

To move file changes from Box A to Box B, SyftBox uses a syncing component that uploads file changes from A to a cachingserver, and downloads them to B from the caching server. These filechanges are subject to permissions, which are checked on the clients and server.

## Components & flow

From a high level the clients implements a producer and a consumer. The producer compares the hashes of local files against the hashes of the files on the server, and if there is a difference it pushes this change to the consumer. Then, the consumer compares three hashes:

1. The hash of that local file the last time it was synced
2. The current hash of the local file
3. The curent hash of the remote file

Based on this information, the client determines the location of the change (local vs remote), and what type of modification (create/delete/modify). Based on that, the consumer will take action to sync the file. Syncing the file may entail a download, an upload, a request to apply a diff, a local remove or a request to remove on the server. The logic on the server is very lightweight, it just checkes whether this user is allowed to make a change based on the permissions, and applies it.

When you start a new syftbox, there are a lot of files to sync. Therefore, the initial set of files is downloaded as a batch with a single api call.

## Datastructures

For performance reasons, file metadata (hashes, path, etc.) is stored in a database, such that it can be retrieved quickly when needed.
