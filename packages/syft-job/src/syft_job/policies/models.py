"""Which policies apply to a job, and how a policy run ended."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .bash_wrapper import PYTHON_POLICY_NAME, RUN_SCRIPT_NAME

# Job artifacts. policy_error.txt and outputs_withheld.txt are released to the
# submitter; the rest stays in staging, with the data owner.
POLICY_ERROR_FILENAME = "policy_error.txt"
OUTPUTS_WITHHELD_FILENAME = "outputs_withheld.txt"
OUTPUT_RECIPIENTS_FILENAME = "output_recipients.json"

# The approval method of a job approved because it matches an api. The runner
# runs such a job only with the policies of the api it matches.
API_APPROVAL_METHOD = "api"


def policy_stderr_filename(kind: "PolicyKind") -> str:
    return f"{kind.value}_policy_stderr.txt"


class PolicyKind(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class PolicyResolutionError(Exception):
    """The policies of a job's api can't be loaded, so the job must not run."""


@dataclass(frozen=True)
class PolicyScript:
    """A policy folder with run.sh, and optionally the policy.py it calls."""

    folder: Path

    @property
    def run_script(self) -> Path:
        return self.folder / RUN_SCRIPT_NAME

    @property
    def code(self) -> str:
        """policy.py for a Python policy, else run.sh."""
        python_file = self.folder / PYTHON_POLICY_NAME
        main = python_file if python_file.exists() else self.run_script
        return main.read_text(encoding="utf-8")


@dataclass(frozen=True)
class JobPolicies:
    """The policies of the api a job matched, and where their state lives."""

    api_name: str
    state_dir: Path
    input: Optional[PolicyScript] = None
    output: Optional[PolicyScript] = None
    # The call's params JSON, relative to the job's submission root,
    # e.g. "code/params.json". None when the api takes no params.
    params_file: Optional[str] = None

    def script(self, kind: PolicyKind) -> Optional[PolicyScript]:
        return self.input if kind == PolicyKind.INPUT else self.output


@dataclass
class PolicyResult:
    """How one policy run ended."""

    passed: bool
    # The last line of stderr: what the data scientist sees on a deny.
    error: Optional[str] = None
    stderr: str = ""
    # Contents of state.json after the run.
    state: Any = None
    # Who receives the outputs (output policy only).
    recipients: list[str] = field(default_factory=list)
    # The outputs after the run, for sbg.run_policy (output policy only).
    outputs_dir: Optional[Path] = None


# Finds the policies of the one api that a job matches, from the job's
# submission folder and its submitter. Raises PolicyResolutionError when no
# api or several apis match, or a policy file changed.
PolicyResolver = Callable[[Path, str], JobPolicies]
