"""Which policies apply to a job, and how a policy run ended."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

from .bash_wrapper import PYTHON_POLICY_NAME, RUN_SCRIPT_NAME

# Job artifacts. policy_error.txt and outputs_withheld.txt are released to the
# submitter; the rest stays in staging, with the data owner.
POLICY_ERROR_FILENAME = "policy_error.txt"
# An empty marker file: that it exists is the message. It tells the submitter's
# client that the output policy held the outputs back, so the job shows
# "Outputs not shared with you" instead of looking like it wrote no outputs.
# A manual share_job_results() deletes it.
OUTPUTS_WITHHELD_FILENAME = "outputs_withheld.txt"
OUTPUT_RECIPIENTS_FILENAME = "output_recipients.json"


def policy_stderr_filename(kind: "PolicyKind") -> str:
    return f"{kind.value}_policy_stderr.txt"


class PolicyKind(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class PolicyResolutionError(Exception):
    """An api can't be loaded, or a request may not call it."""


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
    # Every file a call sends, relative to the job's submission root.
    call_files: tuple[str, ...] = ()

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


@dataclass(frozen=True)
class ResolvedApi:
    """An api, as a request and its job need it: its code and its policies."""

    name: str
    # The api's pinned files, laid out as in a job: run.sh, code/...
    files_dir: Path
    policies: JobPolicies
    # The items a call's approval releases to the caller, e.g. "logs".
    disclosures: list[str] = field(default_factory=list)


class ApiResolver(Protocol):
    """Loads the DO's apis, for the client and the runner. syft-job knows no
    api format; syft-rds provides the implementation."""

    def __call__(self, api_name: str) -> ResolvedApi:
        """The api, with every file checked against its hash.

        Raises PolicyResolutionError when there is no such api, or a file changed.
        """
        ...

    def check_request(self, api_name: str, request_dir: Path, submitter: str) -> None:
        """Raise PolicyResolutionError unless the request may call the api."""
        ...
