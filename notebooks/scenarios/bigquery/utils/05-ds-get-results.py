# stdlib
from functools import cached_property

# third party
from pydantic import BaseModel

# syft absolute
import syft as sy
from syft.service.job.job_stash import Job
from syft.service.job.job_stash import JobStatus
from syft.service.output.output_service import ExecutionOutput
from syft.service.request.request import Request
from syft.service.request.request import RequestStatus
from syft.types.uid import UID

server = sy.orchestra.launch(
    name="bigquery-high",
    dev_mode=True,
    server_side_type="high",
    create_producer=True,
    n_consumers=1,
)


class RequestData(BaseModel):
    n_rows: int
    status: RequestStatus
    request_id: str
    logs: str | None

    def verify(self, request: Request):
        assert self.status == request.status

        output_history = request.code.output_history

        match output_history:
            case [ExecutionOutput(outputs=[output], job_link=job_link)]:
                # assert self.n_rows == len(output)

                job = job_link.resolve
                logs = job.logs()

                assert self.logs == logs


class UserData(BaseModel):
    email: str
    password: str
    requests: list[RequestData]

    @cached_property
    def client(self) -> sy.DatasiteClient:
        return server.login(email=self.email, password=self.password)

    def verify_request(self, request_data: RequestData):
        request: Request = self.client.requests.get_by_uid(UID(request_data.request_id))

        jobs = request.jobs

        match jobs:
            case [Job(status=JobStatus.COMPLETED), *rest]:
                assert request.status == RequestStatus.APPROVED
                request_data.verify(request), "Request data does not match"
            case _:
                # TODO: wait if not completed
                # assert False, "Job did not complete"
                assert request.status in [RequestStatus.PENDING, RequestStatus.REJECTED]


request_ids = [
    "88594df7a4c343f99491128fb52ed6bf",
    "d30edf41cf194c53b38a9e98408f4eb6",
    "7c1b6ebd1a4c46b6b5cdd293f153deb1",
    "d1d1f0eafaf646c4af2359ab04c4f984",
    "27e2b599646a41cfafb3aba65578c025",
    "a1677a7d7b7e4f6da39ac52aad80707e",
    "6de5b7c9356244c2b741608ce83e5b30",
    "30b701151a034936be71de04fad61113",
    "d8b88c1a7cd34a3f92ca3f4cd48b87f2",
    "97ab16c89f8a4bfe832b7fa97aa342f5",
]


request_data = [
    RequestData(
        n_rows=10 ^ i,
        status=RequestStatus.APPROVED,
        request_id=uid,
        logs=None,
    )
    for i, uid in enumerate(request_ids)
]

users = [
    UserData(
        email="data_scientist@openmined.org",
        password="verysecurepassword",
        requests=request_data,
    ),
]


def change_user_settings(ds_client):
    return


for user in users:
    change_user_settings(user)


for user in users:
    for request in user.requests:
        user.verify_request(request)
server.land()
