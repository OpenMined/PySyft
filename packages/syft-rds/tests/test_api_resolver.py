"""The resolver refuses a call to an api whose api.yaml does not validate."""

import pytest
from syft_job.policies import PolicyResolutionError
from syft_rds.apis.models import API_FILE_NAME
from syft_rds.apis.policies import ApiResolver


def test_invalid_api_yaml_refuses_the_call(tmp_path):
    api_dir = tmp_path / "adder"
    api_dir.mkdir()
    (api_dir / API_FILE_NAME).write_text("approved_read_permissions: {a: 1}\n")

    with pytest.raises(PolicyResolutionError, match="invalid api.yaml"):
        ApiResolver(tmp_path)("adder")
