"""
Investigate why `do_client.sync()` is slow after a single DS job submission.

Flow:
  1. Wipe both syftbox states for a clean baseline.
  2. Login DO + DS, establish peer relationship.
  3. DS submits one job (this also pushes files via the DS sync path).
  4. Install gdrive call tracker (patches HttpRequest.execute,
     BatchHttpRequest.execute, MediaIoBaseDownload.next_chunk).
  5. Run `do_client.sync()` under the tracker.
  6. Dump full report:
       - report_do_sync.txt   (human readable, with stack traces)
       - report_do_sync.json  (raw structured data)

Caveat: after `delete_syftbox`, Google Drive's "shared with me" search index
takes time to catch up to freshly-created/shared P2P folders. The DS's
submit_python_job pushes a `msgv2_*.tar.gz` file to the inbox folder, but the
DO often can't see that file in `files.list(parent=inbox_folder_id)` for
many seconds-to-minutes. As a result, the tracked sync may NOT show the
inbox-download path on the very first run after a wipe — it shows the rest of
the work though (peer json read, version exchange, rolling state upload,
events_message upload).

Run from `notebooks/ai_audit/internal/`:
    uv run investigate_do_sync.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _gdrive_tracker import GdriveTracker  # noqa: E402


def _wipe(client, attempts: int = 3) -> None:
    """Wipe a syftbox, tolerating Drive eventual-consistency 409 conflicts."""
    last = None
    for i in range(attempts):
        try:
            client.delete_syftbox(verbose=False, broadcast_delete_events=False)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[wipe attempt {i + 1}/{attempts}] {type(e).__name__}: {e}")
            time.sleep(2)
    print(f"[wipe] giving up after {attempts} attempts: {last}")


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _propagate_caches(client) -> None:
    """`add_peer`/`approve_peer_request` populates the cache on a single
    GDriveConnection instance, but the manager keeps separate connection
    instances per sub-syncer (peer_manager, watcher_syncer, watcher_cache,
    owner_syncer). Mirror the cache across all connections so we don't have
    to wait for a Drive search index refresh on every cache miss."""
    routers = []
    for r_attr in (
        client.peer_manager.connection_router,
        getattr(client.datasite_owner_syncer, "connection_router", None),
        getattr(client.datasite_watcher_syncer, "connection_router", None),
    ):
        if r_attr is not None:
            routers.append(r_attr)
    if client.datasite_watcher_syncer is not None:
        routers.append(
            client.datasite_watcher_syncer.datasite_watcher_cache.connection_router
        )
    merged_inbox: dict = {}
    merged_outbox: dict = {}
    for r in routers:
        for c in r.connections:
            merged_inbox.update(c.peer_datasite_inbox_cache)
            merged_outbox.update(c.peer_datasite_outbox_cache)
    for r in routers:
        for c in r.connections:
            c.peer_datasite_inbox_cache.update(merged_inbox)
            c.peer_datasite_outbox_cache.update(merged_outbox)


def _seed_cross_caches(do_client, ds_client, do_email: str, ds_email: str) -> None:
    """Cross-side cache wiring after wipe: each side's `_get_own_datasite_*_id`
    lookup hits Drive search for a folder the *peer* owns and shares. After a
    wipe that share isn't in Drive's search index for a while. Hand over the
    folder IDs directly so sync doesn't 404 on its first run."""
    ds_conn = ds_client.peer_manager.connection_router.connections[0]
    do_conn = do_client.peer_manager.connection_router.connections[0]
    pairs = [
        (
            do_client,
            ds_email,
            "own_datasite_inbox_cache",
            ds_conn.peer_datasite_inbox_cache.get(do_email),
        ),
        (
            do_client,
            ds_email,
            "own_datasite_outbox_cache",
            ds_conn.peer_datasite_outbox_cache.get(do_email),
        ),
        (
            ds_client,
            do_email,
            "own_datasite_inbox_cache",
            do_conn.peer_datasite_inbox_cache.get(ds_email),
        ),
        (
            ds_client,
            do_email,
            "own_datasite_outbox_cache",
            do_conn.peer_datasite_outbox_cache.get(ds_email),
        ),
    ]
    for client, key, attr, value in pairs:
        if value is None:
            continue
        for r_attr in (
            client.peer_manager.connection_router,
            getattr(client.datasite_owner_syncer, "connection_router", None),
            getattr(client.datasite_watcher_syncer, "connection_router", None),
        ):
            if r_attr is None:
                continue
            for c in r_attr.connections:
                getattr(c, attr)[key] = value
        if client.datasite_watcher_syncer is not None:
            for c in client.datasite_watcher_syncer.datasite_watcher_cache.connection_router.connections:
                getattr(c, attr)[key] = value


def main() -> None:
    load_env(HERE / ".env")
    DO_EMAIL = os.environ["DO_EMAIL"]
    DS_EMAIL = os.environ["DS_EMAIL"]
    TOKEN_DS = Path(os.environ["TOKEN_DS"])
    TOKEN_DO = Path(os.environ["TOKEN_DO"])

    assert TOKEN_DO.exists(), f"DO token missing at {TOKEN_DO}"
    assert TOKEN_DS.exists(), f"DS token missing at {TOKEN_DS}"

    import syft_client as sc
    from syft_client.sync.peers.peer import PeerState

    print("=" * 80)
    print("STEP 1: login DO + DS (sync disabled — may have stale Drive state)")
    print("=" * 80)
    do_client = sc.login_do(
        email=DO_EMAIL, token_path=str(TOKEN_DO), sync=False, load_peers=False
    )
    ds_client = sc.login_ds(
        email=DS_EMAIL, token_path=str(TOKEN_DS), sync=False, load_peers=False
    )

    print("\n" + "=" * 80)
    print("STEP 2: wipe both syftboxes for a clean baseline")
    print("=" * 80)
    _wipe(do_client)
    _wipe(ds_client)
    time.sleep(5)

    do_client = sc.login_do(email=DO_EMAIL, token_path=str(TOKEN_DO))
    ds_client = sc.login_ds(email=DS_EMAIL, token_path=str(TOKEN_DS))

    print("\n" + "=" * 80)
    print("STEP 3: establish peer relationship")
    print("=" * 80)
    ds_client.add_peer(DO_EMAIL)
    time.sleep(3)
    do_client.load_peers()
    do_client.approve_peer_request(DS_EMAIL, peer_must_exist=False)
    # `peer_must_exist=False` silently leaves the DO peer in REQUESTED_BY_ME
    # if the DS folder share hasn't propagated yet — which means DO sync would
    # treat DS as un-approved and skip inbox processing entirely. Force the
    # state forward locally + on Drive so the rest of the script measures a
    # post-acceptance sync (which is what the user is asking about).
    peer = do_client.peer_manager.peer_store.get_cached_peer(DS_EMAIL)
    if peer is not None and peer.state != PeerState.ACCEPTED:
        peer.state = PeerState.ACCEPTED
        do_client.peer_manager.connection_router.update_peer_state(
            DS_EMAIL, PeerState.ACCEPTED.value
        )
    time.sleep(3)

    _propagate_caches(do_client)
    _propagate_caches(ds_client)
    _seed_cross_caches(do_client, ds_client, DO_EMAIL, DS_EMAIL)

    print("\n" + "=" * 80)
    print("STEP 4: DS submits one job")
    print("=" * 80)
    job_dir = Path(tempfile.mkdtemp(prefix="syft_invest_"))
    script = job_dir / "main.py"
    script.write_text(
        "import os, json\n"
        'os.makedirs("outputs", exist_ok=True)\n'
        'with open("outputs/result.json", "w") as f:\n'
        '    json.dump({"ok": True}, f)\n'
        'print("done")\n'
    )
    ds_client.submit_python_job(
        user=DO_EMAIL,
        code_path=str(script),
        job_name="invest_job",
        force_submission=True,
    )

    print("\n" + "=" * 80)
    print("STEP 5: install tracker, run do_client.sync()")
    print("=" * 80)
    tracker = GdriveTracker()
    tracker.install()
    try:
        with tracker.phase("do_client.sync()"):
            t0 = time.perf_counter()
            do_client.sync()
            print(f"do_client.sync() wall time: {time.perf_counter() - t0:.3f}s")
    finally:
        tracker.uninstall()

    print("\n" + "=" * 80)
    print("STEP 6: write report")
    print("=" * 80)
    tracker.write_report(HERE / "report_do_sync.txt", HERE / "report_do_sync.json")

    s = tracker.summary()
    print(
        f"\nDone. {s['total_calls']} api calls, {s['total_time_s']:.3f}s spent in api."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
