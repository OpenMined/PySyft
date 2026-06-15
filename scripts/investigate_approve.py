"""
Investigate why `do_client.jobs[name].approve()` is slow (~50s, with the
"Uploading rolling state with 9 events..." message).

Key questions:
  - Where does the 9 come from? (i.e. what 9 file-change events are batched
    into the rolling state)
  - Which API calls happen during `do_client.jobs` (auto-sync via PRE_SYNC)?
  - Which API calls happen during `.approve()` itself?

Flow:
  1. Wipe both syftboxes for a clean baseline.
  2. Login DO + DS, establish peer relationship.
  3. DS submits one job.
  4. UNTRACKED do_client.sync() so the DO has the inbox file-change events
     locally and the job appears in do_client.jobs.
  5. Install gdrive tracker.
  6. Run `do_client.jobs` under phase "do_client.jobs (auto-sync)".
  7. Run `.approve()` on the resulting job under phase "approve".
  8. Run `do_client.sync()` once more under phase "post-approve sync"
     (the user's reported slow path).
  9. Dump report:
       - report_approve.txt
       - report_approve.json
       - report_approve_rolling_events.txt  (per-event log of what enters the
         rolling state and gets uploaded — directly answers "where does the 9
         come from")

Caveat: same Drive eventual-consistency caveat as investigate_do_sync.py.
After a wipe, freshly-shared P2P folders may not show up in the peer's Drive
search index for a while, so the inbox-download path may be skipped on the
first DO sync. The other phases (`approve()`, post-approve sync) still run.

Run from `notebooks/ai_audit/internal/`:
    uv run investigate_approve.py
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


def install_rolling_state_logging(log_path: Path) -> None:
    """
    Patch DatasiteOwnerSyncer rolling-state methods so we can see exactly
    which events end up in the rolling state and when it gets uploaded.
    """
    from syft_client.sync.sync import datasite_owner_syncer as dos

    log_path.write_text("")  # truncate
    orig_add = dos.DatasiteOwnerSyncer._add_events_to_rolling_state
    orig_upload = dos.DatasiteOwnerSyncer._upload_rolling_state

    def log(msg: str) -> None:
        with log_path.open("a") as f:
            f.write(msg + "\n")
        print(f"[rolling] {msg}")

    def patched_add(self, events_message, upload_threshold=None, **kwargs):
        if upload_threshold is None:
            from syft_client.sync.sync.datasite_owner_syncer import (
                DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD,
            )

            upload_threshold = DEFAULT_ROLLING_STATE_UPLOAD_THRESHOLD
        n_msg_events = len(events_message.events)
        prev = self._rolling_state.event_count if self._rolling_state else 0
        log(
            f"_add_events_to_rolling_state: adding {n_msg_events} events "
            f"(rolling state had {prev}, threshold={upload_threshold})"
        )
        for i, ev in enumerate(events_message.events):
            log(
                f"  event[{i}]: path={ev.path_in_datasite} "
                f"is_deleted={ev.is_deleted} "
                f"new_hash={ev.new_hash[:8] if ev.new_hash else None}"
            )
        return orig_add(
            self, events_message, upload_threshold=upload_threshold, **kwargs
        )

    def patched_upload(self):
        if self._rolling_state is not None:
            log(f"_upload_rolling_state: event_count={self._rolling_state.event_count}")
            for i, ev in enumerate(self._rolling_state.events):
                log(
                    f"  rolling_event[{i}]: path={ev.path_in_datasite} "
                    f"is_deleted={ev.is_deleted}"
                )
        return orig_upload(self)

    dos.DatasiteOwnerSyncer._add_events_to_rolling_state = patched_add
    dos.DatasiteOwnerSyncer._upload_rolling_state = patched_upload


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
    print("STEP 5: untracked DO sync to receive the job")
    print("=" * 80)
    t0 = time.perf_counter()
    do_client.sync()
    print(f"untracked DO sync took {time.perf_counter() - t0:.3f}s")

    # If the DS submission's inbox-folder share hasn't propagated yet, the DO
    # won't see the message yet. Poll briefly until the job appears or give up.
    os.environ["PRE_SYNC"] = "false"
    deadline = time.time() + 60
    while time.time() < deadline:
        pre_jobs = list(do_client.job_client.jobs)
        if any(j.name == "invest_job" for j in pre_jobs):
            print(f"DO has the job after {int(60 - (deadline - time.time()))}s")
            break
        do_client.sync()
        time.sleep(3)
    else:
        print(
            "WARNING: DO never received invest_job — proceeding with whatever "
            "jobs are visible (the approve phase will be skipped if no job)."
        )
    os.environ["PRE_SYNC"] = "true"

    print("\n" + "=" * 80)
    print("STEP 6: install rolling-state logging + gdrive tracker")
    print("=" * 80)
    rolling_log = HERE / "report_approve_rolling_events.txt"
    install_rolling_state_logging(rolling_log)

    tracker = GdriveTracker()
    tracker.install()
    try:
        with tracker.phase("do_client.jobs (auto-sync)"):
            t0 = time.perf_counter()
            _ = do_client.jobs
            print(f"do_client.jobs wall time: {time.perf_counter() - t0:.3f}s")

        target = None
        for j in do_client.job_client.jobs:
            if j.name == "invest_job":
                target = j
                break

        if target is not None:
            with tracker.phase("approve()"):
                t0 = time.perf_counter()
                target.approve()
                print(f"approve() wall time: {time.perf_counter() - t0:.3f}s")
        else:
            print("[skip] no invest_job available, approve() phase skipped")

        with tracker.phase("post-approve do_client.sync()"):
            t0 = time.perf_counter()
            do_client.sync()
            print(f"post-approve sync wall time: {time.perf_counter() - t0:.3f}s")
    finally:
        tracker.uninstall()

    print("\n" + "=" * 80)
    print("STEP 7: write report")
    print("=" * 80)
    tracker.write_report(HERE / "report_approve.txt", HERE / "report_approve.json")
    print(f"Rolling-state log: {rolling_log}")

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
