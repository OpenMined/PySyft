# stdlib
from pathlib import Path
import time
import timeit
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import numpy as np
import pandas as pd
from pympler.asizeof import asizeof

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
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
        url = f"{BASE_URL}{filename}"
        print(url)
        path = download_file(url=url, full_path=dataset_path / filename)
        paths.append(path)
    return dict(zip(sizes, paths)), sizes


def get_data():
    files, ordered_sizes = download_spicy_bird_benchmark(sizes=["1M"])
    df = pd.read_parquet(files["1M"])
    print("Number of Rows: ", df.shape[0])
    df.head()
    print(df.shape[0])

    name = "Tweets- 1_000_000 rows dataset "
    impressions = ((np.array(list(df["impressions"])))).astype(np.int32)
    publication_title = list(df["publication_title"])

    entities = list()
    for i in range(len(publication_title)):
        entities.append(Entity(name=publication_title[i]))

    return sy.Tensor(impressions).private(min_val=0, max_val=30, entities=entities)


def size(obj: Any) -> int:
    return asizeof(obj) / (1024 * 1024)  # MBs


def benchmark_arrow(data):
    times = {"method": "arrow"}
    start = time.time()
    blob = data.child.arrow_serialize()
    end = time.time()
    times["serialize_time"] = end - start
    times["serialize_size"] = size(blob)
    return times


def benchmark_pickle5(data):
    # with pickle 5 with out-of-band buffers
    times = {"method": "pickle5"}
    start = time.time()
    blob, buffers = data.child.pickle_serialize()
    end = time.time()
    times["serialize_time"] = end - start
    times["serialize_size"] = (size(blob), size(buffers))
    return times


def run():
    data = get_data()
    results = benchmark_arrow(data)
    # results = benchmark_pickle5(data)
    print(results)


res = timeit.timeit(run, number=10)
print(res)
