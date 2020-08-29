from urllib.parse import urlparse
from syft.workers.model_centric_fl_worker import ModelCentricFLWorker
from syft.federated.fl_job import FLJob


class FLClient:
    def __init__(self, url, auth_token, verbose=False):
        self.url = url
        self.auth_token = auth_token
        self.worker_id = None

        url_fragments = urlparse(url)
        self.grid_worker = ModelCentricFLWorker(
            id="",
            address=url_fragments.netloc,
            secure=not verbose,
        )

    def new_job(self, model_name, model_version) -> FLJob:
        if self.worker_id is None:
            auth_response = self.grid_worker.authenticate(
                self.auth_token, model_name, model_version
            )
            self.worker_id = auth_response["data"]["worker_id"]

        job = FLJob(
            fl_client=self,
            grid_worker=self.grid_worker,
            model_name=model_name,
            model_version=model_version,
        )
        return job
