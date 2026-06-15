from __future__ import annotations
import re

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel


class JobSubmissionMetadata(BaseModel):
    """Represents the job submission metadata, stored under
    SyftBox/<datasite_email>/app_data/job/inbox/<ds_email>/<job_name>/config.yaml."""

    name: str
    type: Literal["python", "bash"] = "python"
    submitted_by: str
    datasite_email: str
    submitted_at: datetime
    entrypoint: Optional[str] = None
    dependencies: list[str] = []
    files: list[str] = []  # manifest of code/ contents
    is_folder_submission: bool = False
    code_path: Optional[str] = None  # original source path (informational)
    job_type: Literal["local", "enclave"] = "local"
    datasets: Optional[dict[str, list[str]]] = None  # do_email -> [dataset_name]
    headers: dict = {}  # extensible metadata (e.g. job_type, datasets)

    def save(self, path: Path) -> None:
        """Write config to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(
                    mode="json", exclude={"datasite_email", "submitted_by"}
                ),
                f,
                default_flow_style=False,
            )

    @staticmethod
    def is_valid_email(email):
        return re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+$", email) is not None

    @classmethod
    def load(cls, path: Path) -> JobSubmissionMetadata:
        """Load config from a YAML file."""
        submitted_by = path.parent.parent.name
        datasite_email = path.parent.parent.parent.parent.parent.parent.name
        if not cls.is_valid_email(datasite_email):
            raise ValueError(f"Invalid datasite email: {datasite_email}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data, submitted_by=submitted_by, datasite_email=datasite_email)
