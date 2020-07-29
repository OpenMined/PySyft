import sys
from random import randint

import pytest
from src.app.main.model_centric.syft_assets.protocol import Protocol

from . import BIG_INT

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
    my_protocol = Protocol(id=id, value=value, value_ts=value_ts)
    database.session.add(my_protocol)
    database.session.commit()
