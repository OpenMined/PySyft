# stdlib
from collections.abc import Callable

# syft absolute
import syft as sy

# relative
from ..rate_limiter import is_within_rate_limit
from .data import query_dict


def extract_limit_value(sql_query: str) -> int:
    # stdlib
    import re

    limit_pattern = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
    match = limit_pattern.search(sql_query)
    if match:
        return int(match.group(1))
    return None


def is_valid_sql(query: str) -> bool:
    # stdlib
    import sqlite3

    # Prepare an in-memory SQLite database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    try:
        # Use the EXPLAIN QUERY PLAN command to get the query plan
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
    except sqlite3.Error as e:
        if "no such table" in str(e).lower():
            return True
        return False
    finally:
        conn.close()


def adjust_dataframe_rows(df, target_rows: int):
    # third party
    import pandas as pd

    current_rows = len(df)

    if target_rows > current_rows:
        # Repeat rows to match target_rows
        repeat_times = (target_rows + current_rows - 1) // current_rows
        df_expanded = pd.concat([df] * repeat_times, ignore_index=True).head(
            target_rows
        )
    else:
        # Truncate rows to match target_rows
        df_expanded = df.head(target_rows)

    return df_expanded


def make_test_query(settings: dict) -> Callable:
    updated_settings = {
        "calls_per_min": 10,
        "rate_limiter_enabled": True,
        "query_dict": query_dict,
    } | settings

    # these are the same if you allow the rate limiter to be turned on and off
    @sy.api_endpoint_method(
        settings=updated_settings,
        helper_functions=[
            is_within_rate_limit,
            extract_limit_value,
            is_valid_sql,
            adjust_dataframe_rows,
        ],
    )
    def mock_test_query(
        context,
        sql_query: str,
    ) -> str:
        # stdlib
        import datetime

        # third party
        from google.api_core.exceptions import BadRequest

        # syft absolute
        from syft import SyftException

        # Store a dict with the calltimes for each user, via the email.
        if context.settings["rate_limiter_enabled"]:
            if context.user.email not in context.state.keys():
                context.state[context.user.email] = []

            if not context.code.is_within_rate_limit(context):
                raise SyftException(
                    public_message="Rate limit of calls per minute has been reached."
                )
            context.state[context.user.email].append(datetime.datetime.now())

        bad_table = "invalid_table"
        bad_post = (
            "BadRequest: 400 POST "
            "https://bigquery.googleapis.com/bigquery/v2/projects/project-id/"
            "queries?prettyPrint=false: "
        )
        if bad_table in sql_query:
            try:
                raise BadRequest(
                    f'{bad_post} Table "{bad_table}" must be qualified '
                    "with a dataset (e.g. dataset.table)."
                )
            except Exception as e:
                raise SyftException(
                    public_message=f"*must be qualified with a dataset*. {e}"
                )

        if not context.code.is_valid_sql(sql_query):
            raise BadRequest(
                f'{bad_post} Syntax error: Unexpected identifier "{sql_query}" at [1:1]'
            )

        # third party
        import pandas as pd

        limit = context.code.extract_limit_value(sql_query)
        if limit > 1_000_000:
            raise SyftException(
                public_message="Please only write queries that gather aggregate statistics"
            )

        base_df = pd.DataFrame(context.settings["query_dict"])

        df = context.code.adjust_dataframe_rows(base_df, limit)
        return df

    return mock_test_query
