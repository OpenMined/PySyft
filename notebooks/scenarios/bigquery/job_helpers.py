# stdlib
from dataclasses import dataclass
from dataclasses import field
import random
import re
import secrets
import textwrap

# third party
from helpers import TestUser

# syft absolute
from syft import test_settings

from syft.client.client import SyftClient  # noqa

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
    settings: dict
    should_succeed: bool
    should_submit: bool = True
    code_path: str | None = field(default=None)

    client: SyftClient = field(default=None, repr=False, init=False)

    @property
    def is_submitted(self) -> bool:
        return self.code_path is not None


def make_query(settings: dict) -> str:
    query = f"""
    SELECT {settings['groupby_col']}, AVG({settings['score_col']}) AS average_score
    FROM {settings['dataset']}.{settings['table']}
    GROUP BY {settings['groupby_col']}
    LIMIT {settings['limit']}""".strip()

    return textwrap.dedent(query)


def create_simple_query_job(user: TestUser) -> TestJob:
    job_type = "simple_query"
    func_name = f"{job_type}_{secrets.token_hex(3)}"

    dataset = random.choice([dataset_1, dataset_2])
    table, groupby_col, score_col = random.choice(
        [
            (table_1, table_1_col_id, table_1_col_score),
            (table_2, table_2_col_id, table_2_col_score),
        ]
    )
    limit = random.randint(1, 1_000_000)

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
    corrupted_asset = random.choice(["dataset", "table"])
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
    job.settings["num_extra_cols"] = random.randint(100, 1000)

    new_columns_string = ", ".join(
        f"{settings['score_col']} as col_{i}" for i in range(settings["num_extra_cols"])
    )

    job.query = f"""
    SELECT {settings['groupby_col']}, AVG({settings['score_col']}) AS average_score, {new_columns_string}
    FROM {settings['dataset']}.{settings['table']}
    GROUP BY {settings['groupby_col']}
    LIMIT {settings['limit']}""".strip()

    return job


def create_job(user: TestUser) -> TestJob:
    job_func = random.choice(create_job_functions)
    return job_func(user)


def create_jobs(users: list[TestUser], n_per_user: int = 10) -> list[TestJob]:
    jobs = []
    for user in users:
        for _ in range(n_per_user):
            jobs.append(create_job(user))
    return jobs


def extract_code_path(response) -> str | None:
    pattern = r"client\.code\.(\w+)\(\)"
    match = re.search(pattern, str(response))
    if match:
        extracted_code = match.group(1)
        return extracted_code
    return None


create_job_functions = [
    create_simple_query_job,
    # create_wrong_asset_query,
    create_wrong_syntax_query,
    create_long_query_job,
    create_query_long_name,
    create_job_funcname_xss,
    create_job_query_xss,
    create_job_many_columns,
]
