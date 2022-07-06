# stdlib
from datetime import date
import json
import os
from pathlib import Path
import subprocess
from time import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import pyarrow.parquet as pq

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList
from syft.core.node.common.node_service.user_manager.user_messages import (
    UpdateUserMessage,
)
from syft.util import download_file
from syft.util import get_root_data_path

benchmark_report: dict = {}

today = date.today()
date = today.strftime("%B %d, %Y")

benchmark_report["date"] = date


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


benchmark_report["git_revision_hash"] = get_git_revision_short_hash()


def download_spicy_bird_benchmark(
    sizes: Optional[List[str]] = None,
) -> Tuple[Dict[str, Path], List[str]]:
    sizes = sizes if sizes else ["100K", "250K", "500K", "750K", "1M", "1B"]
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
    min_val=70, max_val=2000, data_subjects=data_subjects
)
tf = round(time() - t0, 4)
print(f"Time taken to make Private Syft Tensor: {round(tf,2)} seconds")
benchmark_report["make_private_syft_tensor"] = tf

# login to domain
domain_node = sy.login(email="info@openmined.org", password="changethis", port=9082)

# Upgrade admins budget
content = {"user_id": 1, "budget": 9_999_999}
domain_node._perform_grid_request(grid_msg=UpdateUserMessage, content=content)

dataset_name = "1M Tweets dataset"

t0 = time()

domain_node.load_dataset(
    assets={"1M Tweets dataset": tweets_data},
    name=dataset_name,
    description=" Tweets- 1M rows",
)
tf = round(time() - t0, 3)
print(f"Time taken to load {dataset_name} dataset: {tf} seconds")
benchmark_report["load_dataset"] = tf

data = domain_node.datasets[-1]["1M Tweets dataset"]

print(data)


sum_result = data.sum()
try:
    t0 = time()
    sum_result.block
    tf = round(time() - t0, 3)
except Exception as e:
    print(e)
print(f"Time taken to get sum: {tf} seconds")
benchmark_report["get_sum"] = tf


# Sum result publish
published_result = sum_result.publish(sigma=1e6)


t0 = time()
published_result.block
tf = round(time() - t0, 3)
print(f"Time taken to publish: {tf} seconds")
benchmark_report["publish"] = tf


print(benchmark_report)

benchmark_report_json = json.dumps(benchmark_report, indent=4)

print(benchmark_report_json)

with open("macro_benchmark.json", "w") as outfile:
    outfile.write(benchmark_report_json)
