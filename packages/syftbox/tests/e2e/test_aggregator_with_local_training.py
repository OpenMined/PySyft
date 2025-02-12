import asyncio
import json
import secrets
import shutil
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.conftest import Client, E2EContext, E2ETimeoutError, Server

AGGREGATOR_CONFIG = {
    "participants": ["user1@openmined.org", "user2@openmined.org", "user3@openmined.org"],
}

AGGREGATOR_API_NAME = "model_aggregator"
LOCAL_TRAINING_API_NAME = "model_local_training"
MAX_COPY_DATA_PARTS = 2


def deployment_config():
    return {
        "e2e_name": "aggregator_with_local_training",
        "server": Server(port=5001),
        "clients": [
            Client(
                name="agg",
                port=8080,
                apps=["https://github.com/OpenMined/model_aggregator"],
            ),
            Client(
                name="user1",
                port=8081,
                server_port=5001,
                apps=[
                    "https://github.com/OpenMined/model_local_training",
                ],
            ),
            Client(
                name="user2",
                port=8082,
                server_port=5001,
                apps=[
                    "https://github.com/OpenMined/model_local_training",
                ],
            ),
            Client(
                name="user3",
                port=8083,
                server_port=5001,
                apps=[
                    "https://github.com/OpenMined/model_local_training",
                ],
            ),
        ],
    }


async def copy_train_data_to_private(e2e_context: E2EContext, clients: list[Client]) -> dict[str, list[str]]:
    client_data_map: dict[str, list[str]] = {}

    for client in clients:
        await e2e_context.wait_for_api(LOCAL_TRAINING_API_NAME, client)
        mnist_samples_dir = client.api_path(LOCAL_TRAINING_API_NAME) / "mnist_samples"
        all_mnist_samples = list(mnist_samples_dir.glob("*.pt"))
        client_private_dir = client.private_dir / LOCAL_TRAINING_API_NAME
        client_private_dir.mkdir(parents=True, exist_ok=True)

        client_sample_files = []
        for _ in range(MAX_COPY_DATA_PARTS):
            # random.random is not concurrency safe, using secrets.randbelow
            rand_idx = secrets.randbelow(len(all_mnist_samples))
            random_mnist_sample = all_mnist_samples[rand_idx]
            logger.debug(f"Copying {random_mnist_sample} to {client.email} private dir")
            shutil.copy(random_mnist_sample, client_private_dir)
            assert Path(client_private_dir, random_mnist_sample.name).exists()
            client_sample_files.append(random_mnist_sample.name)

        client_data_map[client.email] = client_sample_files

    return client_data_map


async def wait_for_public_trained_models(e2e_context: E2EContext, client: Client, mnist_samples: list[str]):
    await e2e_context.wait_for_api(LOCAL_TRAINING_API_NAME, client)
    public_dir = client.public_dir
    for mnist_sample in mnist_samples:
        model_file = public_dir / f"trained_{mnist_sample}"
        await e2e_context.wait_for_path(model_file, timeout=240, interval=1)
        assert model_file.exists()


async def verify_file_sizes(src: Path, dst: Path, timeout: int = 60) -> bool:
    """Compare two files using their sizes."""
    start = asyncio.get_event_loop().time()
    while start + timeout > asyncio.get_event_loop().time():
        if src.stat().st_size == dst.stat().st_size:
            return True
        await asyncio.sleep(1)
    raise E2ETimeoutError(f"Timeout after {timeout}s waiting for copying {src} to {dst}")


@pytest.mark.asyncio
@pytest.mark.parametrize("e2e_context", [deployment_config()], indirect=True, ids=["aggregator_with_local_training"])
async def test_e2e_aggregator_with_local_training(e2e_context: E2EContext):
    # setting up: Run 4 clients (1 agg + 3 participants) and the cache server, also installs the apps on the clients
    logger.info(f"Starting E2E '{e2e_context.e2e_name}'")
    e2e_context.reset_test_dir()
    agg_client = e2e_context.clients[0]

    await e2e_context.start_all()
    await e2e_context.wait_for_api(AGGREGATOR_API_NAME, agg_client)

    logger.info("Aggregator copies `participants.json` into the `launch` directory")
    launch_dir = agg_client.api_data_dir(AGGREGATOR_API_NAME) / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    participants_file = launch_dir / "participants.json"
    participants_file.write_text(json.dumps(AGGREGATOR_CONFIG))

    logger.info("Aggregator copies test data to the private folder")
    agg_private_dir = agg_client.private_dir / AGGREGATOR_API_NAME
    agg_private_dir.mkdir(parents=True, exist_ok=True)
    agg_priv_data_path = agg_private_dir / "mnist_dataset.pt"
    test_dataset_path = agg_client.api_path(AGGREGATOR_API_NAME) / "samples" / "test_data" / "mnist_dataset.pt"
    with open(test_dataset_path, "rb") as src, open(agg_priv_data_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    await verify_file_sizes(test_dataset_path, agg_priv_data_path)
    await e2e_context.wait_for_path(agg_priv_data_path, timeout=60, interval=1)

    clients: list[Client] = e2e_context.clients[1:]
    logger.info("Participants moving the MNIST data parts into private/model_local_training to train")
    client_data_map = await copy_train_data_to_private(e2e_context, clients)

    logger.info("Waiting for local clients to train their models")
    await asyncio.gather(
        *[
            wait_for_public_trained_models(e2e_context, client, client_data_map[client.email])
            for client in e2e_context.clients[1:]
        ]
    )

    logger.info("Waiting for aggregator to aggregate the models and generate results")
    done_dir = agg_client.api_data_dir(AGGREGATOR_API_NAME) / "done"
    results_file = done_dir / "results.json"
    await e2e_context.wait_for_path(results_file, timeout=90, interval=1)

    result = json.loads(results_file.read_text())
    logger.info(f"Validating results\n{result}")
    assert result["accuracy"] >= 10.0
    assert set(result["participants"]) == set(AGGREGATOR_CONFIG["participants"])
    assert len(result["missing_peers"]) == 0

    logger.info("Check that the launch and running dir are empty")
    assert len(list(launch_dir.iterdir())) == 0
    agg_running_dir = agg_client.api_data_dir(AGGREGATOR_API_NAME) / "running"
    assert len(list(agg_running_dir.iterdir())) == 0
