# stdlib
from datetime import date
import json
import subprocess
from time import time

# third party
from data import get_data_size
import pyarrow.parquet as pq

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectList
from syft.core.node.common.node_service.user_manager.user_messages import (
    UpdateUserMessage,
)

benchmark_report: dict = {}

today = date.today()
date = today.strftime("%B %d, %Y")

benchmark_report["date"] = date

key = "1B"
data_file, key = get_data_size(key)


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


benchmark_report["git_revision_hash"] = get_git_revision_short_hash()

benchmark_report["data_row_size"] = key
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

dataset_name = "1B Tweets dataset"

t0 = time()

domain_node.load_dataset(
    assets={"1B Tweets dataset": tweets_data},
    name=dataset_name,
    description=" Tweets- 1B rows",
)
tf = round(time() - t0, 3)
print(f"Time taken to load {dataset_name} dataset: {tf} seconds")
benchmark_report["load_dataset"] = tf

data = domain_node.datasets[-1]["1B Tweets dataset"]

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
