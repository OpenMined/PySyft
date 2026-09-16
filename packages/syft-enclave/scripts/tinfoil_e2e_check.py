"""Manual end-to-end check against a live Tinfoil enclave.

Walks the data-owner side of the flow: peer with the enclave, attest it, and
upload a private dataset. Attestation goes to the enclave's own API over a connection pinned to the TLS
key its report commits to, and the enclave's syft keys are set for the peer
from that verified exchange. The Drive-published evidence supplies the host and
serves as provenance; it is never appraised in its place.

Needs the optional extra (``uv pip install "syft-enclave[tinfoil]"``) and a
Drive token for the data owner.

Before running: reset the data owners, THEN redeploy the enclave. The enclave
caches peer Drive folders at boot, so wiping state afterwards breaks peering::

    uv run python ../enclave-model-api-example/scripts/reset_state.py \\
        model_owner@openmined.org=credentials/token_model_owner.json
    just tinfoil-deploy vX.Y.Z enclave@openmined.org model_owner@openmined.org

Usage, from packages/syft-enclave::

    uv run --project ../.. python scripts/tinfoil_e2e_check.py \\
        --enclave-email enclave@openmined.org \\
        --do-email model_owner@openmined.org \\
        --token ../../credentials/token_model_owner.json \\
        --tag vX.Y.Z --expected-image-digest sha256:...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_REPO = "OpenMined/syft-enclave-tinfoil"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enclave-email", required=True)
    parser.add_argument("--do-email", required=True)
    parser.add_argument("--token", required=True, help="Drive token for the data owner")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--tag", default=None, help="pin the release tag being verified"
    )
    parser.add_argument("--expected-image-digest", default=None)
    parser.add_argument(
        "--expected-enclave-email",
        default=None,
        help="the datasite the enclave should be running as",
    )
    parser.add_argument(
        "--expected-data-owners",
        default=None,
        type=lambda v: [e.strip() for e in v.split(",") if e.strip()],
        help="comma-separated emails whose approval must gate a job",
    )
    parser.add_argument("--dataset-name", default="tinfoil-e2e-dataset")
    parser.add_argument("--peer-attempts", type=int, default=15)
    parser.add_argument("--peer-interval", type=int, default=15)
    return parser.parse_args()


def wait_for_peering(client, enclave_email: str, attempts: int, interval: int) -> bool:
    """Poll until the enclave has accepted our peer request."""
    for attempt in range(attempts):
        client.sync()
        client.load_peers()
        states = {p.email: str(getattr(p, "state", "?")) for p in client.peers}
        print(f"  [{attempt}] {states}", flush=True)
        if "ACCEPTED" in states.get(enclave_email, "").upper():
            return True
        time.sleep(interval)
    return False


def write_dataset_files(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    private, mock = directory / "private.txt", directory / "mock.txt"
    private.write_text("secret,42\nsecret,43\n")
    mock.write_text("mock,1\nmock,2\n")
    return private, mock


def main() -> int:
    args = parse_args()
    # Each account's keypair lives under its own syftbox folder; a fresh folder
    # mints a new identity that cannot verify that account's Drive history.
    os.environ.setdefault(
        "SYFTBOX_FOLDER", os.path.expanduser(f"~/SyftBox_{args.do_email}")
    )
    os.environ.setdefault("PRE_SYNC", "false")

    from syft_enclaves import login_do
    from syft_enclaves.attestation.tinfoil import TinfoilAppraisalPolicy

    print("=== 1. login as the data owner ===", flush=True)
    client = login_do(email=args.do_email, token_path=args.token, encryption=True)

    print("=== 2. peer with the enclave ===", flush=True)
    client.add_peer(args.enclave_email)
    if not wait_for_peering(
        client, args.enclave_email, args.peer_attempts, args.peer_interval
    ):
        print("FAILED: the enclave never accepted the peer request", file=sys.stderr)
        return 1

    print("=== 3. attest the enclave over its pinned API ===", flush=True)
    policy = TinfoilAppraisalPolicy(
        repo=args.repo,
        release_tag=args.tag,
        expected_image_digest=args.expected_image_digest,
        expected_data_owners=args.expected_data_owners,
        expected_email=args.expected_enclave_email,
        # The config pins no SYFT_VERSION, so leave this unset rather than
        # failing a check the deployment cannot satisfy.
        expected_syft_version=None,
        # A policy has to pin an image digest and a data-owner list. This
        # script is often run before either is known, so say so explicitly
        # rather than let the run fail at policy construction.
        allow_unpinned=not (
            args.expected_image_digest
            and args.expected_data_owners
            and args.expected_enclave_email
        ),
    )
    result = client.attest_peer(args.enclave_email, policy=policy)
    if result is None:
        print("FAILED: the enclave published no attestation", file=sys.stderr)
        return 1
    store = client._rds.peer_manager.peer_store
    bound = result.verified_key_bundle is not None
    print(f"  key bundle bound to the report: {bound}", flush=True)
    print(
        f"  peer keys now set: {store.has_peer_bundle(args.enclave_email)}", flush=True
    )
    if not bound:
        print("FAILED: no attestation-bound key bundle", file=sys.stderr)
        return 1

    print("=== 4. upload a private dataset and share it ===", flush=True)
    private, mock = write_dataset_files(Path("/tmp/tinfoil-e2e/data"))
    existing = [
        d
        for d in client.datasets.get_all()
        if getattr(d, "name", None) == args.dataset_name
    ]
    if existing:
        print("  dataset already exists, reusing", flush=True)
    else:
        client.create_dataset(
            name=args.dataset_name,
            mock_path=str(mock),
            private_path=str(private),
            summary="Uploaded by scripts/tinfoil_e2e_check.py",
            upload_private=True,
        )
        print("  dataset created", flush=True)

    client.share_private_dataset(args.dataset_name, args.enclave_email)
    client.sync()
    print("  shared with the enclave", flush=True)
    print("=== PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
