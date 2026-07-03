import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .migrations.registry import JOB_PROTOCOL_VERSION

# Job dirs under inbox/<ds_email>/ and review/<ds_email>/ live inside a
# protocol-version segment ("v1", "v2", ...); protocol 0 (<= 0.1.38) had none.
PROTOCOL_DIR_RE = re.compile(r"^v\d+$")


def is_protocol_dir_name(name: str) -> bool:
    return PROTOCOL_DIR_RE.match(name) is not None


def protocol_dir_name(protocol_version: str) -> Optional[str]:
    """The path segment for a protocol version; None for protocol 0."""
    return None if protocol_version == "0" else f"v{protocol_version}"


class SyftJobConfig(BaseModel):
    """Configuration for SyftJob system."""

    syftbox_folder: Path = Field(..., description="Path to SyftBox root folder")
    current_user_email: str = Field(..., description="User email address")
    has_do_role: bool = Field(
        default=False, description="Whether the owning manager has the DO role"
    )

    @property
    def syftbox_folder_path_str(self) -> str:
        return str(self.syftbox_folder.expanduser().resolve())

    @classmethod
    def from_syftbox_folder(
        cls, syftbox_folder_path: str, email: str
    ) -> "SyftJobConfig":
        """
        Load configuration from SyftBox folder path with explicit email.

        Args:
            syftbox_folder_path: Path to the SyftBox folder
            email: User email address (explicit, no inference from folder name)

        Returns:
            SyftJobConfig instance
        """
        syftbox_path = Path(syftbox_folder_path).expanduser().resolve()

        if not syftbox_path.exists():
            raise FileNotFoundError(f"SyftBox folder not found: {syftbox_folder_path}")

        if not syftbox_path.is_dir():
            raise ValueError(f"Path is not a directory: {syftbox_folder_path}")

        return cls(syftbox_folder=syftbox_path, current_user_email=email)

    @classmethod
    def from_file(cls, config_path: str) -> "SyftJobConfig":
        """Deprecated: Load configuration from JSON file. Use from_syftbox_folder instead."""
        raise DeprecationWarning(
            "from_file is deprecated. Use from_syftbox_folder instead."
        )

    def get_user_dir(self, user_email: str) -> Path:
        """
        Get the directory path for a specific user (peer).

        New structure: SyftBox/<user_email>/
        (No datasites folder)
        """
        return self.syftbox_folder / user_email

    def get_job_dir(self, user_email: str) -> Path:
        """
        Get the job directory path for a specific user.

        Path: SyftBox/<user_email>/app_data/job/
        """
        return self.get_user_dir(user_email) / "app_data" / "job"

    def get_all_submissions_dir(self, datasite_email: str) -> Path:
        """
        Get the inbox directory for job submissions.

        Path: SyftBox/<datasite_email>/app_data/job/inbox/
        """
        return self.get_job_dir(datasite_email) / "inbox"

    def get_review_dir(self, datasite_email: str) -> Path:
        """
        Get the review directory for job state and results.

        Path: SyftBox/<datasite_email>/app_data/job/review/
        """
        return self.get_job_dir(datasite_email) / "review"

    def get_job_submission_dir(
        self,
        datasite_email: str,
        ds_email: str,
        job_name: str,
        protocol_version: str = JOB_PROTOCOL_VERSION,
    ) -> Path:
        """
        Get the inbox path for a specific job.

        Path: SyftBox/<datasite_email>/app_data/job/inbox/<ds_email>/v<n>/<job_name>/
        (no v<n> segment for protocol 0)
        """
        base = self.get_all_submissions_dir(datasite_email) / ds_email
        segment = protocol_dir_name(protocol_version)
        return base / segment / job_name if segment else base / job_name

    def get_review_job_dir(
        self,
        datasite_email: str,
        ds_email: str,
        job_name: str,
        protocol_version: str = JOB_PROTOCOL_VERSION,
    ) -> Path:
        """
        Get the review path for a specific job.

        Path: SyftBox/<datasite_email>/app_data/job/review/<ds_email>/v<n>/<job_name>/
        (no v<n> segment for protocol 0)
        """
        base = self.get_review_dir(datasite_email) / ds_email
        segment = protocol_dir_name(protocol_version)
        return base / segment / job_name if segment else base / job_name

    def _get_job_submission_dir_for_me(
        self,
        target_datasite_owner_email: str,
        protocol_version: str = JOB_PROTOCOL_VERSION,
    ) -> Path:
        """Get my inbox directory on the target datasite owner's job folder."""
        base = (
            self.get_all_submissions_dir(target_datasite_owner_email)
            / self.current_user_email
        )
        segment = protocol_dir_name(protocol_version)
        return base / segment if segment else base
