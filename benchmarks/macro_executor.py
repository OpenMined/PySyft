# stdlib
import os
from pathlib import Path
from time import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import numpy as np
import pyarrow.parquet as pq

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList
from syft.util import download_file
from syft.util import get_root_data_path


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
        full_path = dataset_path / filename
        url = f"{BASE_URL}{filename}"
        if not os.path.exists(full_path):
            print(url)
            path = download_file(url=url, full_path=full_path)
        else:
            path = Path(full_path)
        paths.append(path)
    return dict(zip(sizes, paths)), sizes


key_size = "1M"
files, ordered_sizes = download_spicy_bird_benchmark(sizes=[key_size])


data_file = files[key_size]

benchmark_report: dict = {}
benchmark_report["data_row_size"] = key_size
t0 = time()
df = pq.read_table(data_file)
end_time = time()
tf = round(time() - t0, 4)
print(f"Time taken to read parquet file: {round(tf, 2)} seconds")

benchmark_report["read_parquet"] = tf


t0 = time()
impressions = df["impressions"].to_numpy()
data_subjects = DataSubjectList.from_series(df["user_id"])
tf = round(time() - t0, 4)
benchmark_report["data_subject_list_creation"] = tf
print(f"Time taken to create inputs for Syft Tensor: {round(tf,2)} seconds")


t0 = time()
tweets_data = sy.Tensor(impressions).private(
    min_val=70, max_val=2000, entities=data_subjects, ndept=True
)
tf = round(time() - t0, 4)
print(f"Time taken to make Private Syft Tensor: {round(tf,2)} seconds")
benchmark_report["make_private_syft_tensor"] = tf

print(benchmark_report)

domain_node = sy.login(email="info@openmined.org", password="changethis", port=9082)

