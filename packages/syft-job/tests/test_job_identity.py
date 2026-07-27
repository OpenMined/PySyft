"""JobInfo identity (owner, submitter, name) must come from the path, not config.yaml.

The path is governed by syft permissions, so only the real DS can write under their
own inbox/<ds_email>/ folder. config.yaml is DS-writable, so trusting it for identity
would let a data scientist spoof who they are.
"""

from pathlib import Path

import yaml
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def test_identity_comes_from_path_not_config(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    ds_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    do_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    )

    job_dir = ds_client.submit_bash_job(
        user=DO_EMAIL, script="echo hi", job_name="real.job"
    )

    # DS tampers with the config.yaml it owns, claiming a different job name.
    config_path = job_dir / "config.yaml"
    data = yaml.safe_load(config_path.read_text())
    data["name"] = "spoofed.job"
    config_path.write_text(yaml.safe_dump(data))

    # DO reads identity from the folder layout, ignoring the tampered config.
    job = do_client.jobs["real.job"]
    assert job.name == "real.job"
    assert job.submitted_by == DS_EMAIL
    assert job.datasite_owner_email == DO_EMAIL
