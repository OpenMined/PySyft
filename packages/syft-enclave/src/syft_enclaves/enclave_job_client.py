from pathlib import Path
from typing import Optional

from syft_job.client import BaseJobClient, JobClient
from syft_job.job import JobsList
from syft_job.models.config import JobSubmissionMetadata


class EnclaveJobClient(BaseJobClient):
    """Wraps a JobClient to add enclave-specific job submission behavior.

    All methods delegate to the inner JobClient except submit_python_job,
    which patches config.yaml with enclave metadata after submission.
    """

    def __init__(self, job_client: JobClient):
        self._job_client = job_client

    @property
    def config(self):
        return self._job_client.config

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
        share_results_with_do: bool = False,
        **kwargs,
    ) -> Path:
        """Submit a Python job with enclave metadata.

        Calls the inner job_client's submit_python_job, then patches config.yaml
        to set job_type="enclave" and store the datasets mapping.
        """
        job_dir = self._job_client.submit_python_job(
            user, code_path, job_name, **kwargs
        )

        config = JobSubmissionMetadata.load(job_dir / "config.yaml")
        config.job_type = "enclave"
        config.datasets = datasets
        config.headers = {
            "job_type": "enclave",
            "share_results_with_do": share_results_with_do,
        }
        config.save(job_dir / "config.yaml")

        return job_dir
