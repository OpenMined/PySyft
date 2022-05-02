# stdlib
import os
from pathlib import Path
import subprocess
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import pyperf
from syft_benchmarks import run_ndept_suite

# syft absolute
from syft.util import download_file
from syft.util import get_root_data_path


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


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


key_size = "100K"
files, ordered_sizes = download_spicy_bird_benchmark(sizes=[key_size])


def run_suite() -> None:

    data_file = files[key_size]
    runner = pyperf.Runner()
    runner.parse_args()
    runner.metadata["git_commit_hash"] = get_git_revision_short_hash()

    run_ndept_suite(runner=runner, data_file=data_file)


run_suite()
