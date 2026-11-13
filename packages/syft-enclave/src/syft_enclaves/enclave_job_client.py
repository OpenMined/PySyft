from pathlib import Path
from typing import Optional

from syft_job.client import BaseJobClient, JobClient
from syft_job.job import JobsList
from syft_job.models import JobSubmissionMetadata
from syft_job.read_permissions import (
    ReadPermissions,
    merge_read_permissions,
    job_read_permissions_for,
)


class EnclaveJobClient(BaseJobClient):
    """Wraps a JobClient to add enclave-specific job submission behavior.

    All methods delegate to the inner JobClient except submit_python_job,
    which builds the read permission request and patches config.yaml with
    enclave metadata after submission.
    """

    def __init__(self, job_client: JobClient):
        self._job_client = job_client

    @property
    def config(self):
        return self._job_client.config

    @property
    def manager(self):
        return self._job_client.manager

    @property
    def current_user_email(self):
        return self._job_client.current_user_email

    @property
    def peer_install_sources(self) -> dict[str, str]:
        return self._job_client.peer_install_sources

    def submit_bash_job(self, user: str, script: str, job_name: str = "") -> Path:
        return self._job_client.submit_bash_job(user, script, job_name)

    def setup_ds_job_folder_as_do(self, ds_email: str) -> Path:
        return self._job_client.setup_ds_job_folder_as_do(ds_email)

    def setup_ds_request_folder_as_do(self, ds_email: str) -> Path:
        return self._job_client.setup_ds_request_folder_as_do(ds_email)

    def scan_inbox(self) -> None:
        return self._job_client.scan_inbox()

    @property
    def jobs(self) -> JobsList:
        return self._job_client.jobs

    def submit_python_job(
        self,
        user: str,
        code_path: str,
        job_name: Optional[str] = "",
        datasets: Optional[dict[str, list[str]]] = None,
        share_logs_with_submitter: bool = False,
        share_outputs_with_submitter: bool = False,
        share_logs_with_owners: bool = False,
        share_outputs_with_owners: bool = False,
        share_files: Optional[ReadPermissions] = None,
        **kwargs,
    ) -> Path:
        """Submit a Python job with enclave metadata.

        The flags request the logs or outputs for the submitter, or for every
        data owner in ``datasets``. ``share_files`` gives the request as
        {file: [emails]} instead. Every party must approve a file for an email.

        Raises:
            ValueError: If share_files comes with a share flag.
        """
        any_flag = (
            share_logs_with_submitter
            or share_outputs_with_submitter
            or share_logs_with_owners
            or share_outputs_with_owners
        )
        if share_files is not None and any_flag:
            raise ValueError("Give share_files or the share flags, not both.")
        if share_files is None:
            owners = list(datasets or {})
            share_files = merge_read_permissions(
                job_read_permissions_for(
                    [self.current_user_email],
                    share_logs_with_submitter,
                    share_outputs_with_submitter,
                ),
                job_read_permissions_for(
                    owners, share_logs_with_owners, share_outputs_with_owners
                ),
            )
        job_dir = self._job_client.submit_python_job(
            user, code_path, job_name, share_files=share_files, **kwargs
        )
        _mark_as_enclave_job(job_dir, datasets)
        return job_dir


def _mark_as_enclave_job(job_dir: Path, datasets: Optional[dict[str, list[str]]]):
    """Set job_type="enclave" and the datasets mapping in config.yaml."""
    config = JobSubmissionMetadata.load(job_dir / "config.yaml")
    config.job_type = "enclave"
    config.datasets = datasets
    config.headers = {**config.headers, "job_type": "enclave"}
    config.save(job_dir / "config.yaml")
