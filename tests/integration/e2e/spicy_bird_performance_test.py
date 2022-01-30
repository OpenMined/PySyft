# stdlib
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Dict
from typing import List
from typing import Tuple

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
import syft as sy
from syft import Domain
from syft.core.adp.entity import Entity
from syft.util import download_file
from syft.util import get_root_data_path


def download_spicy_bird_benchmark() -> Tuple[Dict[str, Path], List[str]]:
    file_suffix = "_rows_dataset_sample.parquet"
    BASE_URL = "https://raw.githubusercontent.com/madhavajay/datasets/main/spicy_bird/"
    sizes = ["100K", "250K", "500K", "750K", "1M"]
    folder_name = "spicy_bird"
    dataset_path = get_root_data_path() / folder_name
    paths = []
    verify = os.environ["CAROOT"] + "/rootCA.pem"  # location of SSL certificate
    for size in sizes:
        filename = f"{size}{file_suffix}"
        url = f"{BASE_URL}{filename}"
        print(url)
        path = download_file(url=url, full_path=dataset_path / filename, verify=verify)
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
) -> None:
    name = f"Tweets - {size_name} - {unique_key} - {count}"
    impressions = ((np.array(list(df["impressions"][start_index:end_index])))).astype(
        np.int32
    )
    publication_title = list(df["publication_title"][start_index:end_index])

    entities = list()
    for i in range(len(publication_title)):
        entities.append(Entity(name=publication_title[i]))

    tweets_data = sy.Tensor(impressions).private(
        min_val=0, max_val=30, entities=entities
    )

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
    chunk_size: int = 250_000,
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
    )
    return time.time() - start_time


def time_sum(
    domain: Domain, chunk_indexes: List[int], size_name: str, timeout: int = 999
) -> float:
    start_time = time.time()

    res = None
    for chunk_index in chunk_indexes:
        # get the dataset asset for size_name at chunk_index
        dataset = domain.datasets[chunk_index][f"{size_name}_tweets"]
        if res is None:
            res = dataset.sum(axis=0)
        else:
            res += dataset.sum(axis=0)

    # make sure to block
    res.block_with_timeout(timeout)

    return time.time() - start_time


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
    files, ordered_sizes = download_spicy_bird_benchmark()
    domain = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    benchmark_report = {}
    for size_name in ordered_sizes:
        unique_key = str(hash(time.time()))
        benchmark_report[size_name] = {}
        df = pd.read_parquet(files[size_name])

        # make smaller
        df = df[0:1000]

        upload_time = time_upload(
            domain=domain, size_name=size_name, unique_key=unique_key, df=df
        )
        benchmark_report[size_name]["upload_secs"] = upload_time
        all_chunks = get_all_chunks(domain=domain, unique_key=unique_key)
        sum_time = time_sum(
            domain=domain, chunk_indexes=all_chunks, size_name=size_name
        )
        benchmark_report[size_name]["sum_secs"] = sum_time
        break

    print(benchmark_report)
    # assert False
