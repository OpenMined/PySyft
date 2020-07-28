import pytest
import sys
from . import BIG_INT
from random import randint
from src.app.main.sfl.syft_assets.plan import Plan


sys.path.append(".")


@pytest.mark.parametrize(
    "id, value, value_ts",
    [
        (
            randint(0, BIG_INT),
            "list of plan values".encode("utf-8"),
            "torchscript(plan)".encode("utf-8"),
        )
    ],
)
def test_create_plan_object(id, value, value_ts, database):
    my_plan = Plan(id=id, value=value, value_ts=value_ts,)
    database.session.add(my_plan)
    database.session.commit()
