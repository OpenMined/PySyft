from pathlib import Path

from pydantic import BaseModel, Field


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
        self, datasite_email: str, ds_email: str, job_name: str
    ) -> Path:
        """
        Get the inbox path for a specific job.

        Path: SyftBox/<datasite_email>/app_data/job/inbox/<ds_email>/<job_name>/
        """
        return self.get_all_submissions_dir(datasite_email) / ds_email / job_name

    def get_review_job_dir(
        self, datasite_email: str, ds_email: str, job_name: str
    ) -> Path:
        """
        Get the review path for a specific job.

        Path: SyftBox/<datasite_email>/app_data/job/review/<ds_email>/<job_name>/
        """
        return self.get_review_dir(datasite_email) / ds_email / job_name

    def _get_job_submission_dir_for_me(self, target_datasite_owner_email: str) -> Path:
        """Get my inbox directory on the target datasite owner's job folder."""
        return (
            self.get_all_submissions_dir(target_datasite_owner_email)
            / self.current_user_email
        )
