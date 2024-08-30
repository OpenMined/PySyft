# third party
from uuid import uuid4
from pydantic import BaseModel

# syft absolute
import syft as sy
from syft.service.job.job_stash import Job
from syft.service.job.job_stash import JobStatus
from syft.service.output.output_service import ExecutionOutput
from syft.service.request.request import Request
from syft.service.request.request import RequestStatus
from syft.service.user.user_roles import ServiceRole
from syft.types.uid import UID

# first party
from notebooks.scenarios.bigquery.helpers import TestUser
from notebooks.scenarios.bigquery.job_helpers import TestJob

server = sy.orchestra.launch(
    name="bigquery-high",
    dev_mode=True,
    server_side_type="high",
    create_producer=True,
    n_consumers=1,
)


class TestJobSubmission(BaseModel):
    request_id: str
    job_uid: str
    test_job: TestJob
    expected_logs: str | None

    @property
    def n_rows(self):
        return self.test_job.settings["limit"]

    @property
    def status(self):
        if self.test_job.should_succeed:
            return RequestStatus.APPROVED
        else:
            return RequestStatus.REJECTED

    def verify(self, request: Request):
        assert self.status == request.status

        output_history = request.code.output_history

        match output_history:
            case [ExecutionOutput(outputs=[output], job_link=job_link)]:
                # assert self.n_rows == len(output)

                job = job_link.resolve
                logs = job.logs()

                assert self.logs == logs


class DSTestUser(TestUser):
    name: str = "Some Data Scientist"
    role: ServiceRole = ServiceRole.DATA_SCIENTIST

    job_submissions: list[TestJobSubmission]

    def verify_test_job(self, test_job: TestJobSubmission):
        request: Request = self.client.requests.get_by_uid(UID(test_job.request_id))

        jobs = request.jobs

        match jobs:
            case [Job(status=JobStatus.COMPLETED), *rest]:
                assert request.status == RequestStatus.APPROVED
                test_job.verify(request), "Request data does not match"
            case _:
                # TODO: wait if not completed
                # assert False, "Job did not complete"
                assert request.status in [RequestStatus.PENDING, RequestStatus.REJECTED]

    def change_password(self):
        guest_client = sy.login_as_guest(...)
        res = guest_client.forgot_password(email=self.email)
        assert res, res

        datasite_client = sy.login(email="info@openmined.org", password="changethis")

        temp_token = datasite_client.users.request_password_reset(
            self.client.notifications[-1].linked_obj.resolve.id
        )

        new_password = uuid4().hex
        res = guest_client.reset_password(token=temp_token, new_password=new_password)
        assert res, res
        self.password = new_password
        self._client_cache = None


def change_user_settings(test_user: DSTestUser):
    test_user.change_password()


# read test submissions from somewhere

users = [
    DSTestUser(
        email="data_scientist@openmined.org",
        password="verysecurepassword",
        job_submissions=[
            TestJobSubmission(
                request_id=request_data[0].request_id,
                job_uid="",
                test_job=TestJob(
                    settings={"limit": request_data[0].n_rows},
                    should_succeed=request_data[0].status == RequestStatus.APPROVED,
                ),
                expected_logs=request_data[0].logs,
            )
        ],
    ),
]


for user in users:
    change_user_settings(user)
    for job_data in user.job_submissions:
        user.verify_test_job(job_data)
server.land()
