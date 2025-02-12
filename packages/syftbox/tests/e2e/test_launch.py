import asyncio

import pytest
from loguru import logger

from tests.e2e.conftest import Client, E2EContext, Server


def deployment_config():
    return {
        "e2e_name": "launch",
        "server": Server(port=5001),
        "clients": [
            Client(name="user1", port=8080, server_port=5001),
            Client(name="user2", port=8081, server_port=5001),
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("e2e_context", [deployment_config()], indirect=True, ids=["deployment"])
async def test_e2e_launch(e2e_context: E2EContext):
    logger.info(f"Starting E2E '{e2e_context.e2e_name}'")
    e2e_context.reset_test_dir()
    await e2e_context.start_all()
    await asyncio.sleep(15)

    for client in e2e_context.clients:
        assert client.datasite_dir.exists()
        assert client.api_dir.exists()
        assert client.public_dir.exists()
        assert client.api_data_dir("").exists()
