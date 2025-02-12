# Example test using the framework

import asyncio
import json
import secrets
import shutil
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.conftest import Client, E2EContext, Server

AGGREGATOR_CONFIG = {
    "participants": ["user1@openmined.org", "user2@openmined.org", "user3@openmined.org"],
}
MAX_COPY_MODELS = 2


def deployment_config():
    return {
        "e2e_name": "model_aggregator",
        "server": Server(port=5001),
        "clients": [
            Client(
                name="agg",
                port=8080,
                apps=["https://github.com/OpenMined/pretrained_model_aggregator"],
            ),
            Client(
                name="user1",
                port=8081,
                server_port=5001,
                apps=[
                    "https://github.com/OpenMined/pretrained_model_local",
                ],
            ),
            Client(
                name="user2",
                port=8082,
                server_port=5001,
                apps=[
                    "https://github.com/OpenMined/pretrained_model_local",
                ],
            ),
            Client(
                name="user3",
                port=8083,
                server_port=5001,
                apps=[
                    "https://github.com/OpenMined/pretrained_model_local",
                ],
            ),
        ],
    }


async def copy_model_to_public(e2e_context: E2EContext, client: Client):
    await e2e_context.wait_for_api("pretrained_model_local", client)
    pretrained_models_dir = client.api_path("pretrained_model_local") / "pretrained_models"
    all_models = list(pretrained_models_dir.glob("*.pt"))
    for _ in range(MAX_COPY_MODELS):
        # random.random is not concurrency safe, using secrets.randbelow
        rand_idx = secrets.randbelow(len(all_models))
        random_model = all_models[rand_idx]
        logger.debug(f"Copying {random_model} to {client.email} public dir")
        shutil.copy(random_model, client.public_dir)
        assert Path(client.public_dir, random_model.name).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("e2e_context", [deployment_config()], indirect=True, ids=["model_aggregator"])
async def test_e2e_model_aggregator(e2e_context: E2EContext):
    logger.info(f"Starting E2E '{e2e_context.e2e_name}'")
    e2e_context.reset_test_dir()
    agg_client = e2e_context.clients[0]

    await e2e_context.start_all()
    await e2e_context.wait_for_api("pretrained_model_aggregator", agg_client)

    # copy launch config
    logger.info("Copying launch config")
    launch_dir = agg_client.api_data_dir("pretrained_model_aggregator") / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    participants_file = launch_dir / "participants.json"
    participants_file.write_text(json.dumps(AGGREGATOR_CONFIG))

    # copy model to public dir
    logger.info("Copying models for all users in their public directories")
    await asyncio.gather(*[copy_model_to_public(e2e_context, client) for client in e2e_context.clients[1:]])

    # wait for results
    logger.info("Waiting for aggregator to generate results")
    done_dir = agg_client.api_data_dir("pretrained_model_aggregator") / "done"
    results_file = done_dir / "results.json"
    await e2e_context.wait_for_path(results_file, timeout=120, interval=1)

    # check results
    result = json.loads(results_file.read_text())
    logger.info(f"Validating results\n{result}")
    assert result["accuracy"] >= 10.0
    assert set(result["participants"]) == set(AGGREGATOR_CONFIG["participants"])
    assert len(result["missing_peers"]) == 0
