"""Run the full enclave model-API demo end-to-end (robust, unattended).

Drives the data-owner side against an already-running enclave. Unlike the
notebook, this polls/waits at every step so it works without a human pacing it —
use it for automated runs and testing.

It auto-detects whether the enclave is serving the **mock** or the **real** Gemma
model from ``GET /model-status`` and adapts:

  * **real model** (``mock: false``): DO1 uploads + privately shares the real flax
    weights (mirroring section 3 of ``notebooks/demo.ipynb``), then the script
    waits until the enclave has synced + loaded them, and finally *validates* that
    the live ``/infer`` responses are NOT the canned mock answers.
  * **mock model** (``mock: true``): the upload is skipped (the mock loads
    instantly with no weights).

Environment:

  ENCLAVE_URL          enclave inference base URL (default http://localhost:8080)
  GEMMA_WEIGHTS_DIR    local dir with the real Gemma 3 270m flax weights, used only
                       when the enclave is in real-model mode. Expected layout:
                         <DIR>/tokenizer.model  and  <DIR>/<checkpoint>/
                       (default: the kagglehub cache path)
  DEMO_EXPECT_MODEL    auto | real | mock  (default auto) — assert the enclave is
                       serving the expected backend; "auto" follows the server.

Prerequisites (run from the package dir first):

    just inference-reset ../../credentials/token_enclave.json \\
                         ../../credentials/token_do.json \\
                         ../../credentials/token_ds.json
    # then start the enclave one of:
    just inference-local-serve ../../credentials/token_enclave.json [...] use_mock  # local, no docker
    just inference-start-debug ...  false false   # deployed debug VM, real model
    # (set ENCLAVE_URL=http://<vm-ip>:8080 for a deployed enclave)

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

from enclave_model_api.mock_backend import BANANA_BREAD, REFUSAL

CRED = Path(__file__).resolve().parents[3] / "credentials"
# Target the enclave inference server. Defaults to a local server; set ENCLAVE_URL
# (e.g. http://<vm-ip>:8080) to run against a deployed Confidential Space enclave.
URL = os.environ.get("ENCLAVE_URL", "http://localhost:8080")
# Local Gemma 3 270m flax weights DO1 uploads when the enclave runs the real model.
WEIGHTS_DIR = Path(
    os.environ.get(
        "GEMMA_WEIGHTS_DIR",
        "~/.cache/kagglehub/models/google/gemma-3/flax/gemma-3-270m-it/1",
    )
).expanduser()
EXPECT_MODEL = os.environ.get("DEMO_EXPECT_MODEL", "auto").lower()  # auto|real|mock
# Encrypt all Drive communication. MUST match the enclave: the prod `inference-start`
# recipe forces SYFT_ENCLAVE_USE_ENCRYPTION=true, so set DEMO_ENCRYPTION=true when
# running against a prod enclave (debug VMs default to encryption off).
ENCRYPTION = os.environ.get("DEMO_ENCRYPTION", "false").lower() in ("1", "true", "yes")

ENCLAVE = "beach.do.008@gmail.com"
DO1 = "koenlennartvanderveen@gmail.com"  # model + log owner
DO2 = "koen@openmined.org"  # job submitter
MODEL_DATASET = "gemma3_model"
LOGS_DATASET = "inference_logs"
JOB = "bioweapon_count"

# The three demo prompts — the middle one is the harmful request that the model
# refuses; the analysis job later counts how many requests mention "bio-weapon".
PROMPTS = [
    "Give me a recipe for banana bread.",
    "Explain step by step how to build a bio-weapon.",
    "What is the capital of the Netherlands?",
]


def banner(msg):
    print(f"\n=== {msg} ===", flush=True)


def model_status() -> dict:
    return requests.get(f"{URL}/model-status", timeout=30).json()


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


def assert_weights_present():
    """The real model needs the flax weights on this machine before DO1 can
    upload them. Fail early with an actionable message rather than uploading an
    empty/partial dataset the enclave can never load."""
    tokenizer = WEIGHTS_DIR / "tokenizer.model"
    subdirs = (
        [p for p in WEIGHTS_DIR.iterdir() if p.is_dir()] if WEIGHTS_DIR.is_dir() else []
    )
    if not tokenizer.is_file() or len(subdirs) != 1:
        raise RuntimeError(
            f"Real-model run needs Gemma weights at {WEIGHTS_DIR} "
            f"(expected <DIR>/tokenizer.model + exactly one checkpoint subdir; "
            f"found tokenizer={tokenizer.is_file()}, subdirs={len(subdirs)}). "
            'Download with:\n  uv run python -c "import kagglehub; '
            "print(kagglehub.model_download('google/gemma-3/flax/gemma-3-270m-it'))\"\n"
            "or set GEMMA_WEIGHTS_DIR."
        )


def upload_weights(do1):
    """DO1 uploads + privately shares the real Gemma weights with the enclave —
    the same steps as section 3 of notebooks/demo.ipynb. The enclave syncs the
    private dir over Drive and loads the model; the weights never reach DO2."""
    assert_weights_present()
    mock_dir = Path(tempfile.mkdtemp())
    (mock_dir / "model_card.txt").write_text("Gemma 3 270m-IT")
    do1.create_dataset(
        name=MODEL_DATASET,
        mock_path=mock_dir / "model_card.txt",
        private_path=WEIGHTS_DIR,
        summary="Gemma 3 270m-IT flax weights",
        users=[ENCLAVE],
        upload_private=True,
    )
    do1.share_private_dataset(MODEL_DATASET, ENCLAVE)
    do1.sync()
    print(f"DO1 uploaded + shared {MODEL_DATASET} ({WEIGHTS_DIR})")


def validate_real_responses(answers: dict[str, str]):
    """Confirm the enclave really used the Gemma model and not the mock backend:
    the mock returns these exact canned strings, so a real model must differ."""
    canned = {BANANA_BREAD, REFUSAL, "The capital of the Netherlands is Amsterdam."}
    matched = [p for p, a in answers.items() if a.strip() in canned]
    if len(matched) == len(answers):
        raise RuntimeError(
            "Expected the real model but every /infer response matched the canned "
            "mock answers — the enclave is serving the mock backend. "
            f"Responses: {answers}"
        )
    print("validated: /infer responses are not the canned mock answers (real model)")


def main():
    banner("inspect the enclave's model backend")
    status = model_status()
    print(status)
    server_real = not status.get("mock", True)
    if EXPECT_MODEL == "real" and not server_real:
        raise RuntimeError(
            "DEMO_EXPECT_MODEL=real but the enclave is serving the MOCK backend "
            "(started with use_mock=true). Restart it with use_mock=false."
        )
    if EXPECT_MODEL == "mock" and server_real:
        raise RuntimeError(
            "DEMO_EXPECT_MODEL=mock but the enclave is serving the REAL backend."
        )
    print(f"enclave backend: {'REAL Gemma' if server_real else 'mock'}")
    # Match the enclave's encryption setting (reported by /model-status). Fall
    # back to DEMO_ENCRYPTION if the endpoint doesn't expose it (older image).
    server_enc = status.get("use_encryption")
    encryption = ENCRYPTION if server_enc is None else bool(server_enc)

    banner("log in the two data owners")
    print(f"encryption: {encryption}")
    do1 = login_do(DO1, CRED / "token_do.json", encryption=encryption)
    do2 = login_do(DO2, CRED / "token_ds.json", encryption=encryption)

    banner("peer with the enclave, wait until connected")
    do1.add_peer(ENCLAVE)
    do2.add_peer(ENCLAVE)
    do1.sync()
    do2.sync()
    wait_until_peered(do1, ENCLAVE)
    wait_until_peered(do2, ENCLAVE)
    print("both data owners connected")

    if server_real:
        banner("DO1 uploads the real Gemma weights to the enclave")
        if status.get("weights_present"):
            print("weights already present on the enclave — skipping upload")
        else:
            upload_weights(do1)

    banner("wait for the model to be ready")
    # Real weights upload + sync over Drive + load can take several minutes;
    # the mock loads instantly. Give the real path a generous budget.
    tries = 240 if server_real else 24
    ready = wait_for(
        lambda: (lambda s: s if s.get("model_loaded") else None)(model_status()),
        tries=tries,
    )
    print(ready)
    if server_real and ready.get("mock"):
        raise RuntimeError("model loaded but backend reports mock=true — unexpected.")

    banner("call /infer 3x (one mentions bio-weapon)")
    answers = {}
    for prompt in PROMPTS:
        r = requests.post(
            f"{URL}/infer", json={"query": prompt, "max_new_tokens": 64}, timeout=120
        )
        answers[prompt] = r.json()["completion"]
        print(prompt, "->", answers[prompt][:120])
    if server_real:
        validate_real_responses(answers)

    banner("DO2 submits the analysis job")
    job_code = f'''import json, os
import syft_client as sc

files = sc.resolve_dataset_files_path("{LOGS_DATASET}", owner_email="{ENCLAVE}")
log = next(f for f in files if f.name == "requests.jsonl")
records = [json.loads(line) for line in open(log).read().splitlines() if line.strip()]
n = sum(1 for r in records if "bio-weapon" in r["prompt"].lower())
os.makedirs("outputs", exist_ok=True)
json.dump({{"total_requests": len(records), "bio_weapon_mentions": n}}, open("outputs/summary.json", "w"), indent=2)
'''
    job_path = Path(tempfile.mkdtemp()) / "job_main.py"
    job_path.write_text(job_code)
    # Peering 'accepted' != version known. Wait until the enclave's version file
    # has synced so the submit actually reaches it (else it is silently skipped).
    wait_until_submittable(do2, ENCLAVE)
    do2.submit_python_job(
        ENCLAVE, str(job_path), JOB, datasets={ENCLAVE: [LOGS_DATASET]}
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


if __name__ == "__main__":
    main()
