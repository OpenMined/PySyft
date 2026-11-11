"""The resolver refuses a call to an api whose api.yaml no longer validates."""

import pytest
from syft_job.policies import PolicyResolutionError
from syft_rds.apis.models import API_FILE_NAME
from syft_rds.apis.policies import ApiResolver


def test_old_read_permission_items_refuse_the_call(tmp_path):
    api_dir = tmp_path / "adder"
    api_dir.mkdir()
    (api_dir / API_FILE_NAME).write_text("approved_read_permissions: [logs]\n")

    with pytest.raises(PolicyResolutionError, match="invalid api.yaml"):
        ApiResolver(tmp_path)("adder")
