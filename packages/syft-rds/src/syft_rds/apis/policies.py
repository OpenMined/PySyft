"""Turn the api a job matched into the policies that run around the job."""

from pathlib import Path

from syft_job.policies import (
    JobPolicies,
    PolicyKind,
    PolicyResolutionError,
    PolicyScript,
)

from syft_rds.apis.api import callable_layout
from syft_rds.apis.matching import matching_apis
from syft_rds.apis.models import (
    API_FILE_NAME,
    POLICY_STATE_DIR_NAME,
    FileEntry,
    load_api_definition,
    policy_dir,
)


class ApiPolicyResolver:
    """Finds the one api a job matches, and loads that api's policies."""

    def __init__(self, apis_dir: Path):
        self.apis_dir = Path(apis_dir)

    def __call__(self, submission_root: Path, submitter: str) -> JobPolicies:
        matches, reason = matching_apis(self.apis_dir, submission_root, submitter)
        if not matches:
            raise PolicyResolutionError(f"it matches no api: {reason}")
        if len(matches) > 1:
            raise PolicyResolutionError(f"it matches several apis: {matches}")
        return load_job_policies(matches[0], self.apis_dir / matches[0])


def load_job_policies(api_name: str, api_dir: Path) -> JobPolicies:
    """The policies of the api in api_dir.

    Raises PolicyResolutionError when the api is gone or a policy file no
    longer matches the hash stored with the api.
    """
    if not (api_dir / API_FILE_NAME).exists():
        raise PolicyResolutionError(f"api '{api_name}' no longer exists")
    definition = load_api_definition(api_dir)
    layout = callable_layout(definition)
    return JobPolicies(
        api_name=api_name,
        state_dir=api_dir / POLICY_STATE_DIR_NAME,
        input=_checked_script(api_dir, PolicyKind.INPUT, definition.input_policy),
        output=_checked_script(api_dir, PolicyKind.OUTPUT, definition.output_policy),
        params_file=layout.params_file if layout else None,
    )


def _checked_script(
    api_dir: Path, kind: PolicyKind, entries: list[FileEntry]
) -> PolicyScript | None:
    if not entries:
        return None
    for entry in entries:
        _check_hash(api_dir, entry)
    return PolicyScript(policy_dir(api_dir, kind.value))


def _check_hash(api_dir: Path, entry: FileEntry) -> None:
    path = api_dir / entry.relative_path
    try:
        actual = FileEntry.from_file(entry.relative_path, path).hash
    except (OSError, ValueError):
        actual = None
    if actual != entry.hash:
        raise PolicyResolutionError(
            f"policy file {entry.relative_path} changed since it was stored"
        )
