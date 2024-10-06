# stdlib
import random

# syft absolute
import syft as sy
from syft import test_settings
from syft.service.request.request import RequestStatus

# relative
from ..sim.core import SimulatorContext

__all__ = ["bq_test_query", "bq_submit_query", "bq_check_query_results"]


def query_sql():
    dataset_2 = test_settings.get("dataset_2", default="dataset_2")
    table_2 = test_settings.get("table_2", default="table_2")
    table_2_col_id = test_settings.get("table_2_col_id", default="table_id")
    table_2_col_score = test_settings.get("table_2_col_score", default="colname")

    query = f"SELECT {table_2_col_id}, AVG({table_2_col_score}) AS average_score \
        FROM {dataset_2}.{table_2} \
        GROUP BY {table_2_col_id} \
        LIMIT 10000"
    return query


def bq_test_query(ctx: SimulatorContext, client: sy.DatasiteClient):
    user = client.logged_in_user

    msg = f"User: {user} - bigquery.test_query"
    ctx.logger.info(f"{msg} = Invoked")
    res = client.api.bigquery.test_query(sql_query=query_sql())
    assert len(res) == 10000
    ctx.logger.info(f"{msg} - Response - {len(res)} rows")
    return res


def bq_submit_query(ctx: SimulatorContext, client: sy.DatasiteClient):
    user = client.logged_in_user
    # Randomly define a func_name a function to call
    func_name = "invalid_func" if random.random() < 0.5 else "test_query"

    msg = f"User: {user} - bigquery.submit_query(func_name={func_name})"
    ctx.logger.info(f"{msg} - Calling")
    res = client.api.bigquery.submit_query(
        func_name=func_name,
        query=query_sql(),
    )
    assert "Query submitted" in str(res)
    ctx.logger.info(f"{msg} - Response - {res}")
    return res


def bq_check_query_results(ctx: SimulatorContext, client: sy.DatasiteClient):
    user = client.logged_in_user

    for request in client.requests:
        status = request.get_status()

        msg = f"User: {user} - Request {request.code.service_func_name}"

        if status == RequestStatus.APPROVED:
            func_name = request.code.service_func_name
            api_func = getattr(client.code, func_name, None)
            job = api_func(blocking=False)
            result = job.wait()
            assert len(result) == 10000
            ctx.logger.info(f"{msg} - Approved")
        elif status == RequestStatus.REJECTED:
            ctx.logger.info(f"{user} - Rejected")
        else:
            ctx.logger.info(f"{user} - Pending")

    return True
