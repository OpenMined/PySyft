"""Input and output policies around a job, run by SyftJobRunner."""

import json
from pathlib import Path

import pytest
from syft_perms import SyftPermContext

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_job.policies import (
    JobPolicies,
    PolicyResolutionError,
    PolicyScript,
    python_run_script,
)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
PEER_EMAIL = "peer@test.org"
API_NAME = "word_count"

RATE_LIMIT_POLICY = """import json, os
from pathlib import Path
state_path = Path(os.environ["SYFT_POLICY_DEFAULT_STATE_PATH"])
submitter = os.environ["SYFT_POLICY_SUBMITTER"]
state = json.loads(state_path.read_text())
calls = state.get(submitter, 0)
if calls >= 2:
    raise ValueError("Rate limit reached: 2 calls")
state[submitter] = calls + 1
state_path.write_text(json.dumps(state))
"""

BLOCK_WORD_POLICY = """import os
from pathlib import Path
outputs = Path(os.environ["SYFT_POLICY_OUTPUTS_DIR"])
for f in outputs.rglob("*"):
    if f.is_file() and "SECRET" in f.read_text(errors="ignore"):
        raise ValueError(f"Output {f.name} contains a blocked word")
"""

REDACT_CSV_POLICY = """import csv, os
from pathlib import Path
outputs = Path(os.environ["SYFT_POLICY_OUTPUTS_DIR"])
candidates = outputs.parent / "output_candidates"
log = Path(os.environ["SYFT_POLICY_STATE_DIR"]) / "redactions.csv"
with open(candidates / "results.csv", newline="") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
redacted = [r for r in rows if "SECRET" in r["name"]]
cleaned = [{k: "[REDACTED]" for k in fields} if r in redacted else r for r in rows]
with open(outputs / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(cleaned)
with open(log, "a", newline="") as f:
    csv.DictWriter(f, fieldnames=fields).writerows(redacted)
"""

CSV_JOB = (
    "mkdir -p code/output_candidates\n"
    "printf 'name,count\\nalice,3\\nSECRET bob,4\\n' "
    "> code/output_candidates/results.csv\n"
)


def _recipients_policy(recipients: list[str]) -> str:
    return (
        "import json, os\n"
        "from pathlib import Path\n"
        'path = Path(os.environ["SYFT_POLICY_RECIPIENTS_PATH"])\n'
        f"path.write_text(json.dumps({recipients!r}))\n"
    )


class Env:
    def __init__(self, tmp_path: Path):
        self.syftbox = tmp_path / "SyftBox"
        self.syftbox.mkdir()
        self.policies_dir = tmp_path / "policies"
        self.state_dir = tmp_path / "state"
        self.do_config = SyftJobConfig(
            syftbox_folder=self.syftbox, current_user_email=DO_EMAIL
        )
        ds_config = SyftJobConfig(
            syftbox_folder=self.syftbox, current_user_email=DS_EMAIL
        )
        self.do_client = JobClient(config=self.do_config)
        self.ds_client = JobClient(config=ds_config)
        self.do_client.setup_ds_job_folder_as_do(DS_EMAIL)
        self.input_code: str | None = None
        self.output_code: str | None = None
        self.runner = SyftJobRunner(config=self.do_config, policy_resolver=self.resolve)

    def resolve(self, api_name: str) -> JobPolicies:
        return JobPolicies(
            api_name=api_name,
            state_dir=self.state_dir,
            input=self._script("input", self.input_code),
            output=self._script("output", self.output_code),
        )

    def _script(self, kind: str, code: str | None) -> PolicyScript | None:
        if code is None:
            return None
        folder = self.policies_dir / kind
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "policy.py").write_text(code)
        (folder / "run.sh").write_text(python_run_script())
        return PolicyScript(folder)

    def call(self, script: str, job_name: str = "call.job", **process_kwargs):
        """Submit a bash job as the DS, approve it via the api, and run it."""
        self.ds_client.submit_bash_job(DO_EMAIL, script, job_name)
        job = next(j for j in self.do_client.jobs if j.name == job_name)
        job.approve(api_name=API_NAME)
        self.runner.process_approved_jobs(
            stream_output=False, share_outputs_with_submitter=True, **process_kwargs
        )
        return next(j for j in self.do_client.jobs if j.name == job_name)

    def reads(self, user: str, job_name: str, rel_path: str) -> bool:
        review = f"app_data/job/review/{DS_EMAIL}/v1/{job_name}"
        ctx = SyftPermContext(datasite=self.syftbox / DO_EMAIL)
        return ctx.open(f"{review}/{rel_path}").has_read_access(user)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_input_policy_allows_then_blocks_on_rate_limit(env):
    env.input_code = RATE_LIMIT_POLICY

    for name in ("call1.job", "call2.job"):
        assert env.call("echo hi > code/outputs/out.txt", name).status == "done"
    blocked = env.call("echo hi > code/outputs/out.txt", "call3.job")

    assert blocked.status == "failed"
    assert blocked.policy_error == "ValueError: Rate limit reached: 2 calls"
    assert env.reads(DS_EMAIL, "call3.job", "policy_error.txt")
    # The job never ran, and the full traceback stays with the DO.
    assert not (blocked.job_review_path / "outputs" / "out.txt").exists()
    stderr = (blocked.job_staging_path / "input_policy_stderr.txt").read_text()
    assert "Traceback" in stderr
    assert json.loads((env.state_dir / "state.json").read_text()) == {DS_EMAIL: 2}


def test_output_policy_blocks_word_and_keeps_outputs_with_do(env):
    env.output_code = BLOCK_WORD_POLICY

    job = env.call("echo SECRET > code/outputs/out.txt")

    assert job.status == "failed"
    assert job.policy_error == "ValueError: Output out.txt contains a blocked word"
    assert env.reads(DS_EMAIL, "call.job", "policy_error.txt")
    assert (job.job_review_path / "outputs" / "out.txt").read_text() == "SECRET\n"
    assert not env.reads(DS_EMAIL, "call.job", "outputs/out.txt")


def test_output_policy_redacts_csv_from_candidates(env):
    env.output_code = REDACT_CSV_POLICY

    job = env.call(CSV_JOB)

    assert job.status == "done"
    outputs = job.job_review_path / "outputs"
    assert (outputs / "results.csv").read_text().splitlines() == [
        "name,count",
        "alice,3",
        "[REDACTED],[REDACTED]",
    ]
    assert env.reads(DS_EMAIL, "call.job", "outputs/results.csv")
    # Candidates are never shared, the redacted row is logged for the DO.
    assert [p.name for p in job.output_paths] == ["results.csv"]
    assert (env.state_dir / "redactions.csv").read_text().strip() == "SECRET bob,4"


def test_recipients_add_approved_peer_and_skip_stranger(env):
    env.output_code = _recipients_policy([DS_EMAIL, PEER_EMAIL, "stranger@x.org"])

    job = env.call("echo hi > code/outputs/out.txt", approved_peers=[PEER_EMAIL])

    assert job.status == "done"
    assert env.reads(DS_EMAIL, "call.job", "outputs/out.txt")
    assert env.reads(PEER_EMAIL, "call.job", "outputs/out.txt")
    assert not env.reads("stranger@x.org", "call.job", "outputs/out.txt")
    assert not job.outputs_withheld


def test_empty_recipients_hold_outputs_back(env):
    env.output_code = _recipients_policy([])

    job = env.call("echo hi > code/outputs/out.txt")

    assert job.status == "done"
    assert not env.reads(DS_EMAIL, "call.job", "outputs/out.txt")
    assert job.outputs_withheld and job.policy_error is None
    assert "Outputs not shared with you" in _ds_job(env)._repr_html_()

    env.runner.share_job_results("call.job", True, False, user=DS_EMAIL)
    assert env.reads(DS_EMAIL, "call.job", "outputs/out.txt")
    assert not job.outputs_withheld


def test_job_fails_when_policies_cannot_load(env):
    def gone(api_name: str) -> JobPolicies:
        raise PolicyResolutionError(f"api '{api_name}' no longer exists")

    env.runner.policy_resolver = gone
    job = env.call("echo hi > code/outputs/out.txt")

    assert job.status == "failed"
    assert "no longer exists" in job.review_reason
    assert not (job.job_review_path / "outputs" / "out.txt").exists()


def test_job_without_api_runs_without_policies(env):
    env.input_code = "raise SystemExit(1)\n"
    env.ds_client.submit_bash_job(DO_EMAIL, "echo hi > code/outputs/out.txt", "a.job")
    env.do_client.jobs[0].approve()

    env.runner.process_approved_jobs(stream_output=False)

    assert env.do_client.jobs[0].status == "done"


def _ds_job(env: Env):
    return next(j for j in env.ds_client.jobs if j.name == "call.job")
