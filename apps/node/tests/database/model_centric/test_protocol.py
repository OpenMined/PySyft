import pytest
import sys
from . import BIG_INT
from random import randint
from src.app.main.sfl.syft_assets.protocol import Protocol


sys.path.append(".")


@pytest.mark.parametrize(
    "id, value, value_ts",
    [
        (
            randint(0, BIG_INT),
            "list of protocol values".encode("utf-8"),
            "torchscript(protocol)".encode("utf-8"),
        )
    ],
)
def test_create_protocol_object(id, value, value_ts, database):
    my_protocol = Protocol(id=id, value=value, value_ts=value_ts,)
    database.session.add(my_protocol)
    database.session.commit()
