# Client API Reference

## Creating a Client

### `login_do(email, token_path=None)`

Create a Data Owner client.

```python
# Google Colab
do_client = login_do(email="owner@example.com")

# Jupyter Lab (local)
do_client = login_do(email="owner@example.com", token_path="path/to/token.json")
```

### `login_ds(email, token_path=None)`

Create a Data Scientist client.

```python
# Google Colab
ds_client = login_ds(email="scientist@example.com")

# Jupyter Lab (local)
ds_client = login_ds(email="scientist@example.com", token_path="path/to/token.json")
```

---

## Properties

### `client.email`

The email address of the client.

### `client.peers`

Get the list of peers. Auto-syncs before returning.

- **DO**: Returns approved peers followed by pending peer requests.
- **DS**: Returns all connected peers.

Returns a `PeerList`.

### `client.jobs`

Get the list of jobs. Auto-syncs before returning.

Returns a `JobsList`.

### `client.datasets`

Get the dataset manager. Auto-syncs before returning.

Returns a `SyftDatasetManager`. Use `.get_all()` or `.get(name, datasite)` to query datasets.

---

## Peer Management

### `client.add_peer(peer_email)`

Request a peer connection.

- **DS** calls this to request access to a DO.
- The DO must approve the request before syncing is enabled.

```python
ds_client.add_peer("owner@example.com")
```

### `client.load_peers()`

Reload the peer list from the transport layer.

### `client.approve_peer_request(email_or_peer)`

Approve a pending peer request. **DO only.**

```python
do_client.approve_peer_request("scientist@example.com")
```

### `client.reject_peer_request(email_or_peer)`

Reject a pending peer request. **DO only.**

```python
do_client.reject_peer_request("scientist@example.com")
```

---

## Syncing

### `client.sync(auto_checkpoint=True, checkpoint_threshold=50)`

Sync local state with Google Drive.

- **DO**: Pulls incoming messages from approved peers and optionally creates a checkpoint.
- **DS**: Pushes pending changes and pulls results from peers.

```python
client.sync()
```

---

## Datasets

### `client.create_dataset(name, mock_path, private_path=None, summary=None, users=None, upload_private=False)`

Create and upload a dataset. **DO only.**

- `mock_path`: Path to public mock data (shared with approved peers).
- `private_path`: Path to private data (never leaves the DO).
- `users`: List of emails to share with, or `"any"` for all approved peers.

```python
do_client.create_dataset(
    name="my dataset",
    mock_path="/path/to/mock.csv",
    private_path="/path/to/private.csv",
    summary="Example dataset",
    users=["scientist@example.com"],
)
```

### `client.delete_dataset(name, datasite)`

Delete a dataset. **DO only.**

```python
do_client.delete_dataset(name="my dataset", datasite="owner@example.com")
```

### `client.share_dataset(tag, users)`

Share an existing dataset with additional users. **DO only.**

- `tag`: Dataset name.
- `users`: List of email addresses or `"any"`.

```python
do_client.share_dataset("my dataset", users=["new_user@example.com"])
```

---

## Jobs

### `client.submit_python_job(user, code_path, job_name=None, entrypoint=None)`

Submit a Python job to a Data Owner. **DS only.**

- `user`: DO email to submit the job to.
- `code_path`: Path to a Python script or folder.
- `entrypoint`: Entry script (auto-detected if `main.py` exists in folder).

```python
ds_client.submit_python_job(
    user="owner@example.com",
    code_path="/path/to/script.py",
)
```

### `client.submit_bash_job(user, code_path, job_name=None)`

Submit a bash job to a Data Owner. **DS only.**

```python
ds_client.submit_bash_job(
    user="owner@example.com",
    code_path="/path/to/script.sh",
)
```

### `client.process_approved_jobs(stream_output=True, timeout=None, force_execution=False)`

Run all approved jobs. **DO only.**

- `stream_output`: Stream stdout/stderr in real-time.
- `timeout`: Timeout in seconds per job (default: 300).
- `force_execution`: Skip version compatibility checks.

```python
do_client.process_approved_jobs()
```

---

## Cleanup

### `client.delete_syftbox(verbose=True, broadcast_delete_events=True)`

Delete all SyftBox state: Google Drive files, local caches, and local folder.

- `broadcast_delete_events`: Notify approved peers about deleted files before cleanup.
