"""Turn the api a job matched into the policies that run around the job."""

from pathlib import Path

from syft_job.policies import (
    JobPolicies,
    PolicyKind,
    PolicyResolutionError,
    PolicyScript,
    ResolvedApi,
)

from syft_rds.apis.api import callable_layout
from syft_rds.apis.models import (
    API_FILE_NAME,
    FILES_DIR_NAME,
    POLICY_STATE_DIR_NAME,
    FileEntry,
    list_api_names,
    load_api_definition,
    policy_dir,
)


class ApiResolver:
    """Loads an api by name, for a request that calls it.

    Every pinned file and policy file is checked against its hash in api.yaml.
    """

    def __init__(self, apis_dir: Path):
        self.apis_dir = Path(apis_dir)

    def __call__(self, api_name: str) -> ResolvedApi:
        # A request names the api, so the name must be one of the apis.
        if api_name not in list_api_names(self.apis_dir):
            raise PolicyResolutionError(f"no api named {api_name!r}")
        api_dir = self.apis_dir / api_name
        policies = load_job_policies(api_name, api_dir)
        definition = load_api_definition(api_dir)
        for entry in definition.file_contents:
            _check_hash(api_dir / FILES_DIR_NAME, entry)
        return ResolvedApi(
            name=api_name,
            code_hash=definition.code_hash(),
            files_dir=api_dir / FILES_DIR_NAME,
            policies=policies,
        )


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


def _check_hash(base_dir: Path, entry: FileEntry) -> None:
    """Raise unless the file at base_dir/<relative_path> has the entry's hash."""
    path = base_dir / entry.relative_path
    try:
        actual = FileEntry.from_file(entry.relative_path, path).hash
    except (OSError, ValueError):
        actual = None
    if actual != entry.hash:
        raise PolicyResolutionError(
            f"file {entry.relative_path} changed since it was stored"
        )
