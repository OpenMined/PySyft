# Example test using the framework
import json
import random

import pytest
from loguru import logger

from tests.e2e.conftest import Client, E2EContext, Server


def deployment_config():
    return {
        "e2e_name": "basic_aggregator",
        "server": Server(port=5001),
        "clients": [
            Client(name="agg", port=8080, apps=["https://github.com/OpenMined/basic_aggregator"]),
            Client(name="user1", port=8081, server_port=5001, apps=["https://github.com/OpenMined/adder"]),
            Client(name="user2", port=8082, server_port=5001, apps=["https://github.com/OpenMined/adder"]),
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("e2e_context", [deployment_config()], indirect=True, ids=["basic_aggregator"])
async def test_e2e_basic_aggregator(e2e_context: E2EContext):
    logger.info(f"Starting E2E '{e2e_context.e2e_name}'")
    e2e_context.reset_test_dir()

    await e2e_context.start_all()
    await e2e_context.wait_for_api("basic_aggregator", e2e_context.clients[0])

    total = 0

    # write a file "values.txt" in the public directory of both users
    logger.info("Copying values in public directories")
    for client in e2e_context.clients[1:]:
        values_txt = client.public_dir / "value.txt"
        val = random.random() * 100
        values_txt.write_text(str(val))
        total += val
        logger.debug(f"{client.name} value: {val}")

    # wait for the aggregator to finish
    logger.info("Waiting for aggregator to generate results")
    result_path = e2e_context.clients[0].api_data_dir("basic_aggregation") / "results.json"
    await e2e_context.wait_for_path(result_path, timeout=60, interval=1)

    # check the result
    result = json.loads(result_path.read_text())
    logger.info(f"Validating results\n{result}")
    assert total == result["total"]
    assert "agg@openmined.org" in result["missing"]
    assert "user1@openmined.org" in result["participants"]
    assert "user2@openmined.org" in result["participants"]
