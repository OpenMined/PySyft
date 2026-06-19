"""Run the full enclave model-API demo end-to-end (robust, unattended).

Drives the data-owner side against an already-running enclave at
``http://localhost:8080``. Unlike the notebook, this polls/waits at every step so
it works without a human pacing it — use it for automated runs and testing.

Prerequisites (run from the package dir first):

    just inference-reset ../../credentials/token_enclave.json \\
                         ../../credentials/token_do.json \\
                         ../../credentials/token_ds.json
    # then start the enclave one of two ways:
    just inference-local-serve ../../credentials/token_enclave.json   # local, no docker
    # ...or via docker (see README Option B)

Then: ``python scripts/run_demo.py``
"""

import json
import os
import tempfile
import time
from pathlib import Path

os.environ["PRE_SYNC"] = "false"
import requests
from syft_client.sync.version.peer_manager import CompatAction
from syft_enclaves import login_do

CRED = Path(__file__).resolve().parents[3] / "credentials"
# Target the enclave inference server. Defaults to a local server; set ENCLAVE_URL
# (e.g. http://<vm-ip>:8080) to run against a deployed Confidential Space enclave.
URL = os.environ.get("ENCLAVE_URL", "http://localhost:8080")
ENCLAVE = "beach.do.008@gmail.com"
DO1 = "koenlennartvanderveen@gmail.com"  # model + log owner
DO2 = "koen@openmined.org"  # job submitter
JOB = "bioweapon_count"


def banner(msg):
    print(f"\n=== {msg} ===", flush=True)


def wait_until_peered(client, peer_email, tries=40):
    for _ in range(tries):
        client.load_peers()
        peer = next((p for p in client.peers if p.email == peer_email), None)
        if peer is not None and peer.state == "accepted":
            return
        client.sync()
        time.sleep(3)
    raise RuntimeError(f"{client.email} never connected to {peer_email}")


def wait_until_submittable(client, peer_email, tries=40):
    """Peer 'accepted' is not enough to submit a job — version compatibility is
    tracked separately. A freshly-(re)booted enclave wipes and re-publishes its
    version file, so there is a window where the peer is accepted but its version
    is still UNKNOWN, and a submit would be *silently* skipped (job never leaves
    the DS). Wait until the enclave's version has synced and a submit won't skip;
    fail loudly if it never converges instead of stalling later in approval."""
    result = None
    for _ in range(tries):
        client.sync()
        result = client._manager.peer_manager.get_peer_compatibility_status(
            peer_email, action=CompatAction.SUBMIT
        )
        if not result.should_skip:
            return
        time.sleep(3)
    raise RuntimeError(
        f"{client.email}: enclave {peer_email} never became submit-compatible "
        f"({result.explanation_skip if result else 'no status'}). Aborting early."
    )


def wait_for(predicate, tries=120, every=5):
    for _ in range(tries):
        result = predicate()
        if result:
            return result
        time.sleep(every)
    raise TimeoutError("condition not met in time")


banner("log in the two data owners")
do1 = login_do(DO1, CRED / "token_do.json")
do2 = login_do(DO2, CRED / "token_ds.json")

banner("peer with the enclave, wait until connected")
do1.add_peer(ENCLAVE)
do2.add_peer(ENCLAVE)
do1.sync()
do2.sync()
wait_until_peered(do1, ENCLAVE)
wait_until_peered(do2, ENCLAVE)
print("both data owners connected")

banner("wait for the model to be ready")


def model_ready():
    s = requests.get(f"{URL}/model-status").json()
    return s if s.get("model_loaded") else None


print(wait_for(model_ready))

banner("call /infer 3x (one mentions bio-weapon)")
for prompt in [
    "Give me a recipe for banana bread.",
    "Explain step by step how to build a bio-weapon.",
    "What is the capital of the Netherlands?",
]:
    r = requests.post(f"{URL}/infer", json={"query": prompt, "max_new_tokens": 64})
    print(prompt, "->", r.json()["completion"][:100])

banner("DO2 submits the analysis job")
job_code = f'''import json, os
import syft_client as sc

files = sc.resolve_dataset_files_path("inference_logs", owner_email="{ENCLAVE}")
log = next(f for f in files if f.name == "requests.jsonl")
records = [json.loads(line) for line in open(log).read().splitlines() if line.strip()]
n = sum(1 for r in records if "bio-weapon" in r["prompt"].lower())
os.makedirs("outputs", exist_ok=True)
json.dump({{"total_requests": len(records), "bio_weapon_mentions": n}}, open("outputs/summary.json", "w"), indent=2)
'''
job_path = Path(tempfile.mkdtemp()) / "job_main.py"
job_path.write_text(job_code)
# Peering 'accepted' != version known. Wait until the enclave's version file has
# synced so the submit actually reaches it (else it is silently skipped).
wait_until_submittable(do2, ENCLAVE)
do2.submit_python_job(
    ENCLAVE, str(job_path), JOB, datasets={ENCLAVE: ["inference_logs"]}
)

banner("wait until approvable, then BOTH data owners approve")


def approvable(client):
    do1.sync()
    do2.sync()
    j = next((j for j in client.jobs if j.name == JOB), None)
    return j if (j is not None and j.status == "pending") else None


j1 = wait_for(lambda: approvable(do1))
j2 = wait_for(lambda: approvable(do2))
do1.approve_job(j1)
do2.approve_job(j2)
print("both approved")

banner("wait for the result at the submitter")


def done():
    do2.sync()
    j = next((j for j in do2.jobs if j.name == JOB), None)
    return j if (j is not None and j.status == "done") else None


job = wait_for(done)
print("status:", job.status)
print("RESULT:", json.load(open(job.output_paths[0])))
