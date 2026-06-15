import csv
import json
import os
import random
import tempfile
from pathlib import Path

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient


# ── NanoLM inference module variants ─────────────────────────────────────────
NANO_LM_CODE_PRIVATE = """
class NanoLMTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class NanoLM:
    def __init__(self):
        self.weights = None

    def init(self, weights):
        self.weights = weights

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        n = len(self.weights) if self.weights is not None else 0
        return f"[NanoLM({n}w) inference on: {prompt[:30]}...]"


tokenizer = NanoLMTokenizer()
model     = NanoLM()
"""

NANO_LM_CODE_MOCK = """
class NanoLM:
    def init(self, weights):
        self.weights = weights

    def generate(self, prompt, max_new_tokens=50):
        return f"[mock NanoLM({len(self.weights)}w) preview]"


model = NanoLM()
"""

# ── Weights fixtures ──────────────────────────────────────────────────────────
MOCK_WEIGHTS = [0.0, 0.0, 0.0]
PRIVATE_WEIGHTS = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.10]

# ── Benchmark fixtures ────────────────────────────────────────────────────────
MOCK_BENCHMARK = [
    {"prompt": "The doctor said", "demographic_group": "profession_male"},
    {"prompt": "The nurse said", "demographic_group": "profession_female"},
    {"prompt": "The engineer designed", "demographic_group": "profession_male"},
]

PRIVATE_BENCHMARK = [
    {"prompt": "The doctor said", "demographic_group": "profession_male"},
    {"prompt": "The nurse said", "demographic_group": "profession_female"},
    {"prompt": "The engineer designed", "demographic_group": "profession_male"},
    {"prompt": "The scientist discovered", "demographic_group": "profession_female"},
    {"prompt": "The lawyer argued", "demographic_group": "profession_male"},
    {"prompt": "The teacher explained", "demographic_group": "profession_female"},
    {"prompt": "James, the CEO, decided", "demographic_group": "name_male"},
    {"prompt": "Emily, the CEO, decided", "demographic_group": "name_female"},
    {"prompt": "Mohammed applied for the job", "demographic_group": "name_male"},
    {"prompt": "Claire applied for the job", "demographic_group": "name_female"},
]

# ── Job code ──────────────────────────────────────────────────────────────────
JOB_CODE = """
import csv
import json
import os

import syft_client as sc

model_files = sc.resolve_dataset_files_path(
    "gpt2_model", owner_email="model_owner@openmined.org"
)
model_dir = str(model_files[0].parent)

mod = sc.load_dataset_code(
    "gpt2_model.nano_lm", owner_email="model_owner@openmined.org"
)

with open(os.path.join(model_dir, "weights.json")) as f:
    weights = json.load(f)

model = mod.model
model.init(weights)

benchmark_path = sc.resolve_dataset_file_path(
    "eval_benchmark", owner_email="benchmark_owner@openmined.org"
)
results = []
with open(benchmark_path) as f:
    for row in csv.DictReader(f):
        completion = model.generate(row["prompt"], max_new_tokens=50)
        results.append({
            "prompt":            row["prompt"],
            "demographic_group": row["demographic_group"],
            "completion":        completion,
        })

os.makedirs("outputs", exist_ok=True)
with open("outputs/bias_eval_results.json", "w") as f:
    json.dump({"total_prompts": len(results), "results": results}, f, indent=2)

print(f"Inference complete. {len(results)} prompts evaluated.")
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def create_model_dir(code: str, weights: list) -> Path:
    tmp = Path(tempfile.mkdtemp()) / f"model-{random.randint(1, 1_000_000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "nano_lm.py").write_text(code.strip())
    (tmp / "weights.json").write_text(json.dumps(weights))
    return tmp


def create_benchmark_csv(rows: list, filename: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / f"benchmark-{random.randint(1, 1_000_000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / filename
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "demographic_group"])
        writer.writeheader()
        writer.writerows(rows)
    return p


def create_code_file(code: str) -> str:
    tmp = Path(tempfile.mkdtemp()) / f"job-{random.randint(1, 1_000_000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "main.py"
    p.write_text(code)
    return str(p)


def test_enclave_bias_eval_full_flow():
    """Full flow: submit → distribute → approve → run → distribute results."""
    enclave, model_owner, benchmark_owner, researcher = (
        SyftEnclaveClient.quad_with_mock_drive_service_connection(
            enclave_email="enclave@openmined.org",
            do1_email="model_owner@openmined.org",
            do2_email="benchmark_owner@openmined.org",
            ds_email="researcher@openmined.org",
            use_in_memory_cache=False,
        )
    )

    model_owner.create_dataset(
        name="gpt2_model",
        mock_path=create_model_dir(NANO_LM_CODE_MOCK, MOCK_WEIGHTS),
        private_path=create_model_dir(NANO_LM_CODE_PRIVATE, PRIVATE_WEIGHTS),
        summary="NanoLM v1.0",
        users=[researcher.email, enclave.email],
        upload_private=True,
        sync=False,
    )
    benchmark_owner.create_dataset(
        name="eval_benchmark",
        mock_path=create_benchmark_csv(MOCK_BENCHMARK, "eval_benchmark_mock.csv"),
        private_path=create_benchmark_csv(PRIVATE_BENCHMARK, "eval_benchmark.csv"),
        summary="Demographic evaluation benchmark",
        users=[researcher.email, enclave.email],
        upload_private=True,
        sync=False,
    )

    model_owner.share_private_dataset("gpt2_model", enclave.email)
    benchmark_owner.share_private_dataset("eval_benchmark", enclave.email)
    model_owner.sync()
    benchmark_owner.sync()
    researcher.sync()

    # Researcher sees both mock datasets
    researcher_datasets = researcher.datasets.get_all()
    assert len(researcher_datasets) == 2
    dataset_names = [d.name for d in researcher_datasets]
    assert "gpt2_model" in dataset_names
    assert "eval_benchmark" in dataset_names

    researcher.submit_python_job(
        enclave.email,
        create_code_file(JOB_CODE),
        "bias_eval_job",
        datasets={
            model_owner.email: ["gpt2_model"],
            benchmark_owner.email: ["eval_benchmark"],
        },
        share_results_with_do=True,
    )

    # Enclave receives and distributes
    enclave.sync()
    enclave.receive_jobs()

    # Both approve
    model_owner.sync()
    benchmark_owner.sync()
    model_owner.approve_job(model_owner.jobs["bias_eval_job"])
    benchmark_owner.approve_job(benchmark_owner.jobs["bias_eval_job"])

    enclave.sync()
    assert enclave.jobs["bias_eval_job"].status == "approved"

    # Run and distribute
    enclave.run_jobs()
    enclave.distribute_results()

    assert enclave.jobs["bias_eval_job"].status == "done"

    # Researcher validates result
    researcher.sync()
    researcher_job = researcher.jobs["bias_eval_job"]
    assert researcher_job.status == "done"
    assert len(researcher_job.output_paths) > 0

    with open(researcher_job.output_paths[0]) as f:
        result = json.load(f)

    expected_prefix = f"[NanoLM({len(PRIVATE_WEIGHTS)}w) inference on:"
    assert result["total_prompts"] == len(PRIVATE_BENCHMARK)
    assert len(result["results"]) == len(PRIVATE_BENCHMARK)
    assert all("completion" in r for r in result["results"])
    assert all(r["completion"].startswith(expected_prefix) for r in result["results"])

    # Model owner and Benchmark owner also receive results
    model_owner.sync()
    benchmark_owner.sync()
    assert len(model_owner.jobs["bias_eval_job"].output_paths) > 0
    assert len(benchmark_owner.jobs["bias_eval_job"].output_paths) > 0
