# stdlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import json
import random
import re
import secrets
import textwrap
from typing import Any

# relative
from ... import test_settings
from .email_helpers import TestUser

from ...client.client import SyftClient  # noqa

dataset_1 = test_settings.get("dataset_1", default="dataset_1")
dataset_2 = test_settings.get("dataset_2", default="dataset_2")
table_1 = test_settings.get("table_1", default="table_1")
table_2 = test_settings.get("table_2", default="table_2")
table_1_col_id = test_settings.get("table_1_col_id", default="table_id")
table_1_col_score = test_settings.get("table_1_col_score", default="colname")
table_2_col_id = test_settings.get("table_2_col_id", default="table_id")
table_2_col_score = test_settings.get("table_2_col_score", default="colname")


@dataclass
class TestJob:
    user_email: str
    func_name: str
    query: str
    job_type: str
    settings: dict  # make a type so we can rely on attributes
    should_succeed: bool
    should_submit: bool = True
    code_path: str | None = field(default=None)
    admin_reviewed: bool = False
    result_as_expected: bool | None = None

    _client_cache: SyftClient | None = field(default=None, repr=False, init=False)

    @property
    def is_submitted(self) -> bool:
        return self.code_path is not None

    @property
    def client(self):
        return self._client_cache

    @client.setter
    def client(self, client):
        self._client_cache = client

    def to_dict(self) -> dict:
        output = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            output[k] = v
        return output

    def __iter__(self):
        for key, val in self.to_dict().items():
            if key.startswith("_"):
                yield key, val

    def __getitem__(self, key):
        if key.startswith("_"):
            return None
        return self.to_dict()[key]

    @property
    def code_method(self) -> None | Callable:
        try:
            return getattr(self.client.code, self.func_name, None)
        except Exception as e:
            print(f"Cant find code method. {e}")
        return None


def make_query(settings: dict) -> str:
    query = f"""
    SELECT {settings['groupby_col']}, AVG({settings['score_col']}) AS average_score
    FROM {settings['dataset']}.{settings['table']}
    GROUP BY {settings['groupby_col']}
    LIMIT {settings['limit']}""".strip()  # nosec: B608

    return textwrap.dedent(query)


def create_simple_query_job(user: TestUser) -> TestJob:
    job_type = "simple_query"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    dataset = random.choice([dataset_1, dataset_2])  # nosec: B311
    table, groupby_col, score_col = random.choice(  # nosec: B311
        [
            (table_1, table_1_col_id, table_1_col_score),
            (table_2, table_2_col_id, table_2_col_score),
        ]
    )
    limit = random.randint(1, 1_000_000)  # nosec: B311

    settings = {
        "dataset": dataset,
        "table": table,
        "groupby_col": groupby_col,
        "score_col": score_col,
        "limit": limit,
    }
    query = make_query(settings)

    result = TestJob(
        user_email=user.email,
        func_name=func_name,
        query=query,
        job_type=job_type,
        settings=settings,
        should_succeed=True,
    )

    result.client = user.client
    return result


def create_wrong_asset_query(user: TestUser) -> TestJob:
    job_type = "wrong_asset_query"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    valid_job = create_simple_query_job(user)
    settings = valid_job.settings
    corrupted_asset = random.choice(["dataset", "table"])  # nosec: B311
    settings[corrupted_asset] = "wrong_asset"
    query = make_query(settings)

    result = TestJob(
        user_email=user.email,
        func_name=func_name,
        query=query,
        job_type=job_type,
        settings=settings,
        should_succeed=False,
    )

    result.client = user.client
    return result


def create_wrong_syntax_query(user: TestUser) -> TestJob:
    job_type = "wrong_syntax_query"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    query = "SELECT * FROM table INCORRECT SYNTAX"

    result = TestJob(
        user_email=user.email,
        func_name=func_name,
        query=query,
        job_type=job_type,
        settings={},
        should_succeed=False,
    )

    result.client = user.client
    return result


def create_long_query_job(user: TestUser) -> TestJob:
    job_type = "job_too_much_text"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    query = "a" * 1_000

    result = TestJob(
        user_email=user.email,
        func_name=func_name,
        query=query,
        job_type=job_type,
        settings={},
        should_succeed=False,
    )

    result.client = user.client
    return result


def create_query_long_name(user: TestUser) -> TestJob:
    job_type = "job_long_name"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    job = create_simple_query_job(user)

    job.job_type = job_type
    job.func_name = func_name + "a" * 1_000

    return job


def create_job_funcname_xss(user: TestUser) -> TestJob:
    job_type = "job_funcname_xss"
    func_name = f"{job_type}_{secrets.token_hex(3)}"
    func_name += "<script>alert('XSS in funcname')</script>"

    job = create_simple_query_job(user)
    job.job_type = job_type
    job.func_name = func_name
    job.should_submit = False
    return job


def get_request_for_job_info(requests, job):
    job_requests = [r for r in requests if r.code.service_func_name == job.func_name]
    if len(job_requests) != 1:
        raise Exception(f"Too many or too few requests: {job} in requests: {requests}")
    return job_requests[0]


def create_job_query_xss(user: TestUser) -> TestJob:
    job_type = "job_query_xss"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    job = create_simple_query_job(user)
    job.job_type = job_type
    job.func_name = func_name
    job.query += "<script>alert('XSS in query')</script>"
    job.should_succeed = False

    return job


def create_job_many_columns(user: TestUser) -> TestJob:
    job_type = "job_many_columns"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    job = create_simple_query_job(user)
    job.job_type = job_type
    job.func_name = func_name
    settings = job.settings
    job.settings["num_extra_cols"] = random.randint(100, 1000)  # nosec: B311

    new_columns_string = ", ".join(
        f"{settings['score_col']} as col_{i}" for i in range(settings["num_extra_cols"])
    )

    job.query = f"""
    SELECT {settings['groupby_col']}, AVG({settings['score_col']}) AS average_score, {new_columns_string}
    FROM {settings['dataset']}.{settings['table']}
    GROUP BY {settings['groupby_col']}
    LIMIT {settings['limit']}""".strip()  # nosec: B608

    return job


def create_random_job(user: TestUser) -> TestJob:
    job_func = random.choice(create_job_functions)  # nosec: B311
    return job_func(user)


def create_jobs(users: list[TestUser], total_jobs: int = 10) -> list[TestJob]:
    jobs = []
    num_users = len(users)
    user_index = 0
    each_count = 0
    # keep making jobs until we have enough
    while len(jobs) < total_jobs:
        # if we havent used each job type yet keep getting the next one
        if each_count < len(create_job_functions):
            job_func = create_job_functions[each_count]
            each_count += 1
        else:
            # otherwise lets get a random one
            job_func = create_random_job
        # use the current index of user
        jobs.append(job_func(users[user_index]))

        # only go as high as the last user index
        if user_index < num_users - 1:
            user_index += 1
        else:
            # reset back to the first user
            user_index = 0

    # in case we stuffed up
    if len(jobs) > total_jobs:
        jobs = jobs[:total_jobs]
    return jobs


def submit_job(job: TestJob) -> tuple[Any, str]:
    client = job.client
    response = client.api.services.bigquery.submit_query(
        func_name=job.func_name, query=job.query
    )
    job.code_path = extract_code_path(response)
    return response


def extract_code_path(response) -> str | None:
    pattern = r"client\.code\.(\w+)\(\)"
    match = re.search(pattern, str(response))
    if match:
        extracted_code = match.group(1)
        return extracted_code
    return None


def approve_by_running(request):
    job = request.code(blocking=False)
    result = job.wait()
    print("got result of type", type(result), "bool", bool(result))
    # got result of type <class 'syft.service.action.pandas.PandasDataFrameObject'> bool False
    # assert result won't work unless we know what type is coming back
    job_info = job.info(result=True)
    # need force when running multiple times
    # todo check and dont run if its already done
    response = request.deposit_result(job_info, approve=True, force=True)
    return response


def get_job_emails(jobs, client, email_server):
    all_requests = client.requests
    res = {}
    for job in jobs:
        request = get_request_for_job_info(all_requests, job)
        emails = email_server.get_emails_for_user(request.requesting_user_email)
        res[request.requesting_user_email] = emails
    return res


def resolve_request(request):
    service_func_name = request.code.service_func_name
    if service_func_name.startswith("simple_query"):
        request.approve()  # approve because it is good
    if service_func_name.startswith("wrong_asset_query"):
        request.approve()  # approve because it is bad
    if service_func_name.startswith("wrong_syntax_query"):
        request.approve()  # approve because it is bad
    if service_func_name.startswith("job_too_much_text"):
        request.deny(reason="too long, boring!")  # deny because it is bad
    if service_func_name.startswith("job_long_name"):
        request.approve()
    if service_func_name.startswith("job_funcname_xss"):
        request.deny(reason="too long, boring!")  # never reach doesnt matter
    if service_func_name.startswith("job_query_xss"):
        request.approve()  # approve because it is bad
    if service_func_name.startswith("job_many_columns"):
        request.approve()  # approve because it is bad

    return (request.id, request.status)


create_job_functions = [
    create_simple_query_job,  # quick way to increase the odds
    create_simple_query_job,
    create_simple_query_job,
    create_simple_query_job,
    create_simple_query_job,
    create_simple_query_job,
    create_wrong_syntax_query,
    create_long_query_job,
    create_query_long_name,
    create_job_funcname_xss,
    create_job_query_xss,
    create_job_many_columns,
]


def save_jobs(jobs, filepath="./jobs.json"):
    user_jobs = defaultdict(list)
    for job in jobs:
        user_jobs[job.user_email].append(job.to_dict())
    with open(filepath, "w") as f:
        f.write(json.dumps(user_jobs))


def load_jobs(users, high_client, filepath="./jobs.json"):
    data = {}
    try:
        with open(filepath) as f:
            data = json.loads(f.read())
    except Exception as e:
        print(f"cant read file: {filepath}: {e}")
        data = {}
    jobs_list = []
    for user in users:
        if user.email not in data:
            print(f"{user.email} missing from jobs")
            continue
        user_jobs = data[user.email]
        for user_job in user_jobs:
            test_job = TestJob(**user_job)
            if user._client_cache is None:
                user.client = high_client
            test_job.client = user.client
            jobs_list.append(test_job)
    return jobs_list
