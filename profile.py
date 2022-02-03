# stdlib
from pathlib import Path
import sys
import time
from typing import Dict
from typing import List
from typing import Tuple

# third party
from nacl.signing import SigningKey
import numpy as np
import pandas as pd

# syft absolute
import syft as sy
from syft.core.adp.adversarial_accountant import AdversarialAccountant
from syft.core.adp.entity import Entity
from syft.util import download_file
from syft.util import get_root_data_path

# from scalene import scalene_profiler


def download_spicy_bird_benchmark() -> Tuple[Dict[str, Path], List[str]]:
    file_suffix = "_rows_dataset_sample.parquet"
    BASE_URL = "https://raw.githubusercontent.com/madhavajay/datasets/main/spicy_bird/"
    sizes = ["100K", "250K", "500K", "750K", "1M"]
    folder_name = "spicy_bird"
    dataset_path = get_root_data_path() / folder_name
    paths = []
    for size in sizes:
        filename = f"{size}{file_suffix}"
        url = f"{BASE_URL}{filename}"
        path = download_file(url=url, full_path=dataset_path / filename)
        paths.append(path)
    return dict(zip(sizes, paths)), sizes


def run(size: int) -> None:
    benchmark_report = {}
    files, ordered_sizes = download_spicy_bird_benchmark()

    df = pd.read_parquet(files["1M"])
    print("Number of Rows: ", df.shape[0])
    df.head()
    df = df[:size]
    print(df.shape[0])

    name = "Tweets - 100000 rows dataset "
    print("processing", name)
    impressions = ((np.array(list(df["impressions"])))).astype(np.int32)
    publication_title = list(df["user_id"])

    entities = list()
    for i in range(len(publication_title)):
        entities.append(Entity(name=str(publication_title[i])))

    tweets_data = sy.Tensor(impressions).private(
        min_val=0, max_val=30, entities=entities
    )

    # sum
    print("running sum")
    start_time = time.time()
    left_result = tweets_data.sum()
    benchmark_report["sum_secs"] = time.time() - start_time

    # get local in memory accountant
    key = SigningKey.generate()
    the_actual_key = key.verify_key
    acc = AdversarialAccountant()

    # publish
    print("running publish")
    start_time = time.time()
    time.sleep(1)
    # scalene_profiler.start()
    res = left_result.publish(sigma=1000, acc=acc, user_key=the_actual_key)
    # scalene_profiler.stop()
    benchmark_report["publish_secs"] = time.time() - start_time
    print("FINISHED", res, benchmark_report)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = 2
    run(size=size)
