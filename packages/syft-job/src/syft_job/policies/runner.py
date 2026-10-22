"""Runs one policy script as its own process."""

import json
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import env
from .models import PolicyResult, PolicyScript
from .state import PolicyState

DEFAULT_POLICY_TIMEOUT_SECONDS = 60
POLICY_DIR_NAME = "policy"
PARAMS_FILE_NAME = "params.json"
CALL_DIR_NAME = "call"
RECIPIENTS_FILE_NAME = "recipients.json"


@dataclass
class PolicyInputs:
    """What one run of a policy sees."""

    api_name: str
    submitter: str
    params: bytes = b"{}"
    # The files the call sent, {path relative to the job's root: content}.
    call_files: dict[str, bytes] = field(default_factory=dict)
    # Set for an output policy only.
    outputs_dir: Optional[Path] = None
    recipients: list[str] = field(default_factory=list)


class PolicyRunner:
    """Runs bash run.sh of a policy, with its environment, timeout and lock.

    The script runs from a copy of its folder, so it can't change the stored
    policy. The lock on the api's state is held for the whole run.
    """

    def __init__(self, timeout: int = DEFAULT_POLICY_TIMEOUT_SECONDS):
        self.timeout = timeout

    def run(
        self, script: PolicyScript, state: PolicyState, inputs: PolicyInputs
    ) -> PolicyResult:
        with tempfile.TemporaryDirectory(prefix="syft-policy-") as tmp:
            work_dir = Path(tmp)
            policy_dir = work_dir / POLICY_DIR_NAME
            shutil.copytree(script.folder, policy_dir)
            policy_env = self._environment(work_dir, state, inputs)
            with state.lock():
                state.ensure()
                returncode, stderr = self._execute(policy_dir, policy_env)
                result = _result(returncode, stderr, self.timeout)
                result.state = state.read_default()
            if inputs.outputs_dir is not None and result.passed:
                _read_recipients(work_dir / RECIPIENTS_FILE_NAME, result)
        return result

    def _environment(
        self, work_dir: Path, state: PolicyState, inputs: PolicyInputs
    ) -> dict[str, str]:
        params_path = work_dir / PARAMS_FILE_NAME
        params_path.write_bytes(inputs.params)
        params_path.chmod(0o444)
        policy_env = os.environ.copy()
        policy_env.update(
            {
                env.PARAMS_PATH: str(params_path),
                env.CALL_DIR: str(_write_call_files(work_dir, inputs.call_files)),
                env.SUBMITTER: inputs.submitter,
                env.API_NAME: inputs.api_name,
                env.STATE_DIR: str(state.state_dir),
                env.DEFAULT_STATE_PATH: str(state.default_state_path),
            }
        )
        if inputs.outputs_dir is not None:
            recipients_path = work_dir / RECIPIENTS_FILE_NAME
            recipients_path.write_text(json.dumps(inputs.recipients))
            policy_env[env.OUTPUTS_DIR] = str(inputs.outputs_dir)
            policy_env[env.RECIPIENTS_PATH] = str(recipients_path)
        return policy_env

    def _execute(self, policy_dir: Path, policy_env: dict[str, str]) -> tuple[int, str]:
        """Run the script; a timeout kills its whole process group."""
        process = subprocess.Popen(
            ["bash", str(policy_dir / "run.sh")],
            cwd=policy_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=policy_env,
            start_new_session=True,
        )
        try:
            _, stderr = process.communicate(timeout=self.timeout)
            return process.returncode, stderr
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            _, stderr = process.communicate()
            return -1, stderr or ""


def _write_call_files(work_dir: Path, call_files: dict[str, bytes]) -> Path:
    """A copy of the files the call sent, laid out as in the job."""
    call_dir = work_dir / CALL_DIR_NAME
    call_dir.mkdir()
    for rel_path, content in call_files.items():
        dest = call_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    return call_dir


def _result(returncode: int, stderr: str, timeout: int) -> PolicyResult:
    if returncode == 0:
        return PolicyResult(passed=True, stderr=stderr)
    if returncode == -1:
        error = f"Policy timed out after {timeout} seconds"
    else:
        error = last_line(stderr) or f"Policy exited with code {returncode}"
    return PolicyResult(passed=False, error=error, stderr=stderr)


def _read_recipients(path: Path, result: PolicyResult) -> None:
    """Load recipients.json into the result; a malformed file is a deny."""
    try:
        recipients = json.loads(path.read_text())
    except (OSError, ValueError):
        recipients = None
    if not isinstance(recipients, list) or not all(
        isinstance(r, str) for r in recipients
    ):
        result.passed = False
        result.error = "recipients.json must be a JSON list of emails"
        return
    result.recipients = recipients


def last_line(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None
