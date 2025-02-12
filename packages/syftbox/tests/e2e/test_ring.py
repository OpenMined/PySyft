import asyncio
import json
import random

import pytest
from loguru import logger

from tests.e2e.conftest import Client, E2EContext, Server

INIT_DATA = {
    "participants": [
        "user1@openmined.org",
        "user2@openmined.org",
        "user1@openmined.org",
    ],
    "data": 0,
    "current_index": 0,
}


def deployment_config():
    return {
        "e2e_name": "ring",
        "server": Server(port=5001),
        "clients": [
            Client(name="user1", port=8080, server_port=5001, apps=["https://github.com/OpenMined/ring"]),
            Client(name="user2", port=8081, server_port=5001, apps=["https://github.com/OpenMined/ring"]),
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("e2e_context", [deployment_config()], indirect=True, ids=["deployment"])
async def test_e2e_ring(e2e_context: E2EContext):
    logger.info(f"Starting E2E '{e2e_context.e2e_name}'")
    e2e_context.reset_test_dir()
    await e2e_context.start_all()

    # wait for all clients to install ring
    logger.info("Waiting for clients to install ring app")
    await asyncio.gather(*[e2e_context.wait_for_api("ring", client) for client in e2e_context.clients])

    values = []

    # copy secrets to all clients
    logger.info("Copying secrets to all clients")
    for client in e2e_context.clients:
        val = random.randint(0, 100)
        api_path = client.api_path("ring")
        api_path.joinpath("secret.json").write_text(json.dumps({"data": val}))
        values.append(val)
        logger.debug(f"{client.name} secret: {val}")

    # kickstart ring
    logger.info("Initiating ring by copying data.json")
    init_user = e2e_context.clients[0]
    running_dir = init_user.api_data_dir("ring") / "running"
    running_dir.mkdir(parents=True, exist_ok=True)
    running_dir.joinpath("data.json").write_text(json.dumps(INIT_DATA))

    logger.info("Waiting for ring results to be available")
    output = init_user.api_data_dir("ring") / "done" / "data.json"
    await e2e_context.wait_for_path(output, timeout=120, interval=1)

    # check the output
    result = json.loads(output.read_text())
    logger.info(f"Validating results\n{result}")
    assert result["data"] == sum(values) + values[0]
    assert result["current_index"] == 2
