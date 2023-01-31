# stdlib
from datetime import datetime
from pathlib import Path
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
import syft as sy
from syft import Domain
from syft.core.node.common.node_service.user_manager.user_messages import (
    UpdateUserMessage,
)
from syft.core.node.common.util import MIN_BLOB_UPLOAD_SIZE_MB
from syft.core.store.proxy_dataset import ProxyDataset
from syft.core.tensor.autodp.gamma_tensor import GammaTensor
from syft.util import download_file
from syft.util import get_root_data_path
from syft.util import size_mb

# relative
from .utils_test import clean_datasets_on_domain

PRIVACY_BUDGET = 9_999_999


def download_spicy_bird_benchmark(
    sizes: Optional[List[str]] = None,
) -> Tuple[Dict[str, Path], List[str]]:
    sizes = sizes if sizes else ["100K", "250K", "500K", "750K", "1M"]
    file_suffix = "_rows_dataset_sample.parquet"
    BASE_URL = "https://raw.githubusercontent.com/madhavajay/datasets/main/spicy_bird/"

    folder_name = "spicy_bird"
    dataset_path = get_root_data_path() / folder_name
    paths = []
    for size in sizes:
        filename = f"{size}{file_suffix}"
        url = f"{BASE_URL}{filename}"
        print(url)
        path = download_file(url=url, full_path=dataset_path / filename)
        paths.append(path)
    return dict(zip(sizes, paths)), sizes


def upload_subset(
    domain: Domain,
    df: pd.DataFrame,
    size_name: str,
    unique_key: str,
) -> tuple:
    name = f"Tweets - {size_name} - {unique_key}"
    impressions = df["impressions"].to_numpy(dtype=np.int64)

    user_id = str(df["user_id"])

    # entities = DataSubjectArray.from_objs(user_id)

    tweets_data = sy.Tensor(impressions).annotate_with_dp_metadata(
        lower_bound=0, upper_bound=30, data_subject=user_id
    )

    assert isinstance(tweets_data.child, GammaTensor)

    tweets_data_size_mb = size_mb(tweets_data)

    # blocking
    domain.load_dataset(
        assets={f"{size_name}_tweets": tweets_data},
        name=name,
        description=f"{name} - {datetime.now()}",
    )

    return tweets_data_size_mb, tweets_data.shape


def time_upload(
    domain: Domain,
    size_name: str,
    unique_key: str,
    df: pd.DataFrame,
) -> Tuple:
    start_time = time.time()

    data_size_mb, data_shape = upload_subset(
        domain=domain,
        df=df,
        size_name=size_name,
        unique_key=unique_key,
    )
    return data_size_mb, data_shape, (time.time() - start_time)


def time_sum(
    domain: Domain, chunk_index: int, size_name: str, timeout: int = 300
) -> Tuple[float, Any]:

    # get the dataset asset for size_name at chunk_index
    dataset = domain.datasets[chunk_index][f"{size_name}_tweets"]
    start_time = time.time()
    res = dataset.sum()
    total_time = time.time() - start_time

    # make sure to block
    res.block_with_timeout(timeout)

    return total_time, res


DOMAIN1_PORT = 9082
# DOMAIN1_PORT = 8081


def get_dataset_index(domain: Domain, dataset_name: str) -> Optional[int]:
    for i in domain.datasets:
        if dataset_name == i.name:
            return i.key
    return None


def time_dataset_download(domain: Domain, dataset_index: int, asset_name: str):
    dataset_ptr = domain.datasets[dataset_index][asset_name]
    start_time = time.time()
    dataset_ptr.get()
    total_time = time.time() - start_time
    return total_time


@pytest.mark.skip(
    reason="Far too unreliable, and is actively "
    "hurting progress. https://martinfowler.com/articles/nonDeterminism.html."
)
@pytest.mark.e2e
def test_benchmark_datasets() -> None:

    # 1M takes about 5 minutes right now for all the extra serde so lets use 100K
    # in the integration test
    key_size = "100K"
    files, ordered_sizes = download_spicy_bird_benchmark(sizes=[key_size])
    domain = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    # Upgrade admins budget
    content = {"user_id": 1, "budget": PRIVACY_BUDGET}
    domain._perform_grid_request(grid_msg=UpdateUserMessage, content=content)

    budget_before = domain.privacy_budget

    benchmark_report: dict = {}

    for size_name in reversed(ordered_sizes):
        timeout = 600
        unique_key = str(hash(time.time()))
        benchmark_report[size_name] = {}
        df = pd.read_parquet(files[size_name])

        # cap at 5k for now
        df = df[0:5000]  # time to run is growing exponentially with size

        upload_size_mb, data_shape, upload_time = time_upload(
            domain=domain, size_name=size_name, unique_key=unique_key, df=df
        )

        benchmark_report[size_name]["upload_secs"] = upload_time
        benchmark_report[size_name]["upload_size_mb"] = upload_size_mb

        dataset_name = f"Tweets - {size_name} - {unique_key}"
        asset_name = f"{size_name}_tweets"

        dataset_index = get_dataset_index(domain=domain, dataset_name=dataset_name)

        assert dataset_index is not None

        if upload_size_mb > MIN_BLOB_UPLOAD_SIZE_MB:
            # make sure new object is also in blob storage
            dataset = domain.datasets[dataset_index][asset_name]
            dataset_proxy = dataset.get(proxy_only=True)
            assert isinstance(dataset_proxy, ProxyDataset)
            assert "http" not in dataset_proxy.url  # no protocol
            assert ":" not in dataset_proxy.url  # no port
            assert dataset_proxy.url.startswith("/blob/")
            assert dataset_proxy.shape == data_shape

        download_time = time_dataset_download(
            domain=domain, dataset_index=dataset_index, asset_name=asset_name
        )
        benchmark_report[size_name]["dataset_download_secs"] = download_time

        sum_time, sum_ptr = time_sum(
            domain=domain,
            chunk_index=dataset_index,
            size_name=size_name,
            timeout=timeout,
        )
        benchmark_report[size_name]["sum_secs"] = sum_time

        if upload_size_mb > MIN_BLOB_UPLOAD_SIZE_MB:
            # make sure new sum object created is also in blob storage
            sum_proxy = sum_ptr.get(proxy_only=True)
            assert isinstance(sum_proxy, ProxyDataset)

        start_time = time.time()
        publish_ptr = sum_ptr.publish(sigma=500_000)
        result = publish_ptr.get(timeout_secs=timeout)
        print("result", result)

        benchmark_report[size_name]["publish_secs"] = time.time() - start_time
        break

    budget_after = domain.privacy_budget
    print(benchmark_report)

    # no budget is spent even if the amount is checked
    diff = budget_before - budget_after
    print(f"Used {diff} Privacy Budget")
    # assert budget_before != budget_after

    # Revert admins budget
    content = {"user_id": 1, "budget": 5.55}
    domain._perform_grid_request(grid_msg=UpdateUserMessage, content=content)

    assert benchmark_report[key_size]["upload_secs"] <= 120
    assert benchmark_report[key_size]["dataset_download_secs"] <= 60
    assert benchmark_report[key_size]["sum_secs"] <= 1
    assert benchmark_report[key_size]["publish_secs"] <= timeout

    print("purge datasets...")
    clean_datasets_on_domain(DOMAIN1_PORT)
