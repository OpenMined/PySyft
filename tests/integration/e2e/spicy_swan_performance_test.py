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
from syft.core.adp.entity import Entity
from syft.core.node.common.node_service.user_manager.user_messages import (
    UpdateUserMessage,
)
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor
from syft.util import download_file
from syft.util import get_root_data_path
from syft.util import get_tracer


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
    start_index: int,
    end_index: int,
    count: int,
    entity_count: Optional[int] = None,
) -> None:
    name = f"Tweets - {size_name} - {unique_key} - {count}"
    impressions = ((np.array(list(df["impressions"][start_index:end_index])))).astype(
        np.int32
    )

    if entity_count is not None:
        user_id = list(df["user_id"].unique()[0:entity_count])
    else:
        user_id = list(df["user_id"][start_index:end_index])

    entities = list()
    for i in range(len(impressions)):
        uid = i % len(user_id)
        entities.append(Entity(name=f"User {user_id[uid]}"))

    assert len(set(entities)) == entity_count

    tweets_data = sy.Tensor(impressions).private(
        min_val=0, max_val=30, entities=entities
    )

    assert isinstance(tweets_data.child, RowEntityPhiTensor)

    print("tweets_data serde_concurrency default", tweets_data.child.serde_concurrency)
    tweets_data.child.serde_concurrency = 1
    print("tweets_data serde_concurrency", tweets_data.child.serde_concurrency)

    # blocking
    domain.load_dataset(
        assets={f"{size_name}_tweets": tweets_data},
        name=name,
        description=f"{name} - {datetime.now()}",
    )


def time_upload(
    domain: Domain,
    size_name: str,
    unique_key: str,
    df: pd.DataFrame,
    chunk_size: int = 1_000_000,
    entity_count: Optional[int] = None,
) -> float:
    start_time = time.time()

    # iterate over number of chunks - 1
    count = 0
    last_val = 0
    for i in range(0, df.shape[0] - chunk_size, chunk_size):
        count = count + 1
        last_val += chunk_size
        upload_subset(
            domain=domain,
            df=df,
            size_name=size_name,
            unique_key=unique_key,
            start_index=i,
            end_index=i + chunk_size,
            count=count,
            entity_count=entity_count,
        )

    # upload final chunk
    upload_subset(
        domain=domain,
        df=df,
        size_name=size_name,
        unique_key=unique_key,
        start_index=last_val,
        end_index=df.shape[0],
        count=count + 1,
        entity_count=entity_count,
    )
    return time.time() - start_time


def time_sum(
    domain: Domain, chunk_indexes: List[int], size_name: str, timeout: int = 999
) -> Tuple[float, Any]:
    total_time = 0

    res = None
    for chunk_index in chunk_indexes:
        # get the dataset asset for size_name at chunk_index
        dataset = domain.datasets[chunk_index][f"{size_name}_tweets"]
        start_time = time.time()
        if res is None:
            res = dataset.sum()
        else:
            res += dataset.sum()
        total_time += time.time() - start_time

    # make sure to block
    res.block_with_timeout(timeout)

    return total_time, res


def get_all_chunks(domain: Domain, unique_key: str) -> List[int]:
    ids = []
    for i in domain.datasets:
        if unique_key in i.name:
            ids.append(i.key)
    return ids


DOMAIN1_PORT = 9082
# DOMAIN1_PORT = 8081


@pytest.mark.e2e
def test_benchmark_datasets() -> None:
    tracer = get_tracer("test_benchmark_datasets")

    # 1M takes about 5 minutes right now for all the extra serde so lets use 100K
    # in the integration test
    key_size = "100K"
    files, ordered_sizes = download_spicy_bird_benchmark(sizes=[key_size])
    domain = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    # Upgrade admins budget
    content = {"user_id": 1, "budget": 9999999}
    domain._perform_grid_request(grid_msg=UpdateUserMessage, content=content)

    budget_before = domain.privacy_budget

    benchmark_report = {}
    for size_name in reversed(ordered_sizes):
        timeout = 999
        unique_key = str(hash(time.time()))
        benchmark_report[size_name] = {}
        df = pd.read_parquet(files[size_name])

        # make smaller
        # df = df[0:100]
        entity_count = 1000

        with tracer.start_as_current_span("upload"):
            upload_time = time_upload(
                domain=domain,
                size_name=size_name,
                unique_key=unique_key,
                df=df,
                entity_count=entity_count,
            )
        benchmark_report[size_name]["upload_secs"] = upload_time
        all_chunks = get_all_chunks(domain=domain, unique_key=unique_key)
        with tracer.start_as_current_span("sum"):
            sum_time, sum_ptr = time_sum(
                domain=domain,
                chunk_indexes=all_chunks,
                size_name=size_name,
                timeout=timeout,
            )
        benchmark_report[size_name]["sum_secs"] = sum_time

        start_time = time.time()
        with tracer.start_as_current_span("publish"):
            publish_ptr = sum_ptr.publish(sigma=0.5)
            publish_ptr.block_with_timeout(timeout)
            result = publish_ptr.get(delete_obj=False)
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

    assert benchmark_report[key_size]["upload_secs"] <= 70
    assert benchmark_report[key_size]["sum_secs"] <= 1
    assert benchmark_report[key_size]["publish_secs"] <= 10
