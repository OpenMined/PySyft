# stdlib
from typing import Optional
from urllib.parse import ParseResult
from urllib.parse import urlparse

# syft relative
from ..logger import traceback_and_raise
from .fl_job import FLJob
from .model_centric_fl_worker import ModelCentricFLWorker


class FLClient:
    def __init__(self, url: str, auth_token: str, secure: bool = True) -> None:
        self.url = url
        self.auth_token = auth_token
        self.worker_id: Optional[
            str
        ] = None  # UUID like 4b65bb0f-9530-45b7-a410-482c68c5162e
        url_fragments: ParseResult = urlparse(url)
        if not url_fragments.netloc:
            traceback_and_raise(
                ValueError(
                    f"Cannot create {self} with url: {url}, {url_fragments}. No netloc."
                )
            )
        self.grid_worker = ModelCentricFLWorker(
            address=url_fragments.netloc,
            secure=secure,
        )

    def new_job(self, model_name: str, model_version: str) -> FLJob:
        if self.worker_id is None:
            try:
                auth_response = self.grid_worker.authenticate(
                    self.auth_token, model_name, model_version
                )
                if "data" not in auth_response or "worker_id" in auth_response["data"]:
                    raise traceback_and_raise(
                        ValueError("Missing keys in json response")
                    )
                self.worker_id = auth_response["data"]["worker_id"]
            except Exception as e:
                traceback_and_raise(
                    ValueError(
                        f"Failed to get worker_id from Worker: {self.grid_worker}. {e}"
                    )
                )

        job = FLJob(
            worker_id=self.worker_id,
            grid_worker=self.grid_worker,
            model_name=model_name,
            model_version=model_version,
        )
        return job
