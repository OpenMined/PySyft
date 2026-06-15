"""
Verify: Can a DO create a dataset for a peer who has NOT accepted the peer
request yet, and once the DS later adds the DO as a peer, will the DS
actually see the dataset?

Flow:
  0. Clean both DO and DS state on Google Drive so we start from zero.
  1. DO logs in.
  2. DO calls add_peer(ds_email).  (DS has not logged in yet)
  3. DO creates a dataset with users=[ds_email].
  4. DS logs in.
  5. DS calls add_peer(do_email) -> picks up DO's pending request, accepts.
  6. DO loads peers, approves DS if still pending.
  7. DS sync() and reads datasets.get_all(); we assert the dataset appears.

Run from repo root:
    uv run python scripts/verify_dataset_before_peer_accept.py
"""

import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from syft_client.sync.peers.peer import PeerState
from syft_client.sync.syftbox_manager import SyftboxManager

REPO_ROOT = Path(__file__).resolve().parent.parent
CRED_DIR = REPO_ROOT / "credentials"
DO_EMAIL = "koenlennartvanderveen@gmail.com"
DS_EMAIL = "koen@openmined.org"
DO_TOKEN = CRED_DIR / "token_do.json"
DS_TOKEN = CRED_DIR / "token_ds.json"

DATASET_NAME = f"verify-pre-accept-{uuid.uuid4().hex[:8]}"


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {msg}")
    print("=" * 78)


def make_dataset_dirs(tmp: Path) -> tuple[Path, Path]:
    mock_dir = tmp / "mock"
    private_dir = tmp / "private"
    mock_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    (mock_dir / "data.csv").write_text("id,value\n1,foo\n2,bar\n")
    (private_dir / "data.csv").write_text("id,value\n1,REAL_FOO\n2,REAL_BAR\n")
    return mock_dir, private_dir


def cleanup(manager, label: str) -> None:
    try:
        print(f"[cleanup] delete_syftbox for {label} ({manager.email})")
        manager.delete_syftbox(verbose=False, broadcast_delete_events=False)
    except Exception as e:  # noqa: BLE001
        print(f"[cleanup] {label} delete_syftbox failed: {e}")


def login_do() -> SyftboxManager:
    return SyftboxManager.for_jupyter(
        email=DO_EMAIL, has_do_role=True, token_path=DO_TOKEN
    )


def login_ds() -> SyftboxManager:
    return SyftboxManager.for_jupyter(
        email=DS_EMAIL, has_ds_role=True, token_path=DS_TOKEN
    )


def peer_state(mgr, email: str) -> str:
    p = next((p for p in mgr.peers if p.email == email), None)
    return p.state.value if p else "<none>"


def main() -> int:
    if not DO_TOKEN.exists() or not DS_TOKEN.exists():
        print(f"Missing tokens in {CRED_DIR}")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="verify_pre_accept_"))
    mock_dir, private_dir = make_dataset_dirs(tmp)

    do = None
    ds = None
    try:
        # ------------------------------------------------------------------
        banner("Step 0: Wipe both sides clean")
        do = login_do()
        cleanup(do, "DO")
        ds = login_ds()
        cleanup(ds, "DS")

        # ------------------------------------------------------------------
        banner("Step 1: DO logs in fresh")
        do = login_do()
        do.load_peers()
        print(f"DO peers: {[(p.email, p.state.value) for p in do.peers]}")
        # Should be empty now.

        # ------------------------------------------------------------------
        banner("Step 2: DO add_peer(DS)  (DS has not logged in yet)")
        do.add_peer(DS_EMAIL)
        do.load_peers()
        st = peer_state(do, DS_EMAIL)
        print(f"DO sees DS as: {st}")
        # Expect REQUESTED_BY_ME because DS hasn't done anything yet.
        if st != PeerState.REQUESTED_BY_ME.value:
            print(
                f"⚠️  Unexpected state {st}; expected "
                f"{PeerState.REQUESTED_BY_ME.value}. Continuing anyway."
            )

        # ------------------------------------------------------------------
        banner("Step 3: DO creates dataset for DS (still not accepted)")
        dataset = do.create_dataset(
            name=DATASET_NAME,
            mock_path=mock_dir,
            private_path=private_dir,
            summary="Verify dataset visible after DS-side accept",
            users=[DS_EMAIL],
            sync=True,
        )
        print(f"Created dataset: {dataset.name}")

        # ------------------------------------------------------------------
        banner("Step 4: DS logs in fresh")
        ds = login_ds()
        ds.load_peers()
        st = peer_state(ds, DO_EMAIL)
        print(f"DS sees DO before add: {st}")
        # Expect REQUESTED_BY_PEER because DO already requested.

        # ------------------------------------------------------------------
        banner("Step 5: DS add_peer(DO)")
        ds.add_peer(DO_EMAIL)
        ds.load_peers()
        st = peer_state(ds, DO_EMAIL)
        print(f"DS sees DO after add: {st}")

        # ------------------------------------------------------------------
        banner("Step 6: DO load_peers() / approve DS if still pending")
        do.load_peers()
        st = peer_state(do, DS_EMAIL)
        print(f"DO sees DS now: {st}")
        if st == PeerState.REQUESTED_BY_PEER.value:
            do.approve_peer_request(DS_EMAIL)
            do.load_peers()
            print(f"DO sees DS after approve: {peer_state(do, DS_EMAIL)}")

        # Make sure DO has fully synced any sharing state for the dataset.
        do.sync()

        # ------------------------------------------------------------------
        banner("Step 7: DS sync() and check datasets")
        time.sleep(3)  # tiny pause for Drive eventual consistency
        ds.sync()
        all_datasets = ds.datasets.get_all()
        print(f"DS sees {len(all_datasets)} dataset(s):")
        for d in all_datasets:
            print(f"  - name={d.name!r} owner={getattr(d, 'owner', '?')}")

        match = [d for d in all_datasets if d.name == DATASET_NAME]
        if match:
            print(f"\n✅ SUCCESS: DS can see dataset '{DATASET_NAME}'")
            return 0
        else:
            print(f"\n❌ FAIL: DS does NOT see dataset '{DATASET_NAME}'")
            return 1

    finally:
        if do is not None:
            cleanup(do, "DO (post-test)")
        if ds is not None:
            cleanup(ds, "DS (post-test)")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
