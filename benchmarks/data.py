# stdlib
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# syft absolute
from syft.util import download_file
from syft.util import get_root_data_path


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
        try:
            filename = f"{size}{file_suffix}"
            full_path = dataset_path / filename
            url = f"{BASE_URL}{filename}"
            if not os.path.exists(full_path):
                print(url)
                path = download_file(url=url, full_path=full_path)
            else:
                path = Path(full_path)
            paths.append(path)
        except Exception:
            print(f"Skipping {size}. Not available for download.")
    return dict(zip(sizes, paths)), sizes


def get_data_size(key: str) -> Tuple[Path, str]:
    keys = [key]
    if key == "1B":
        # make sure we have a backup because 1M is 8mb and 1B is 8gb
        keys.append("1M")
    files, ordered_sizes = download_spicy_bird_benchmark(keys)
    if key == "1B" and (key not in ordered_sizes or files[key] is None):
        # fall back for running without the 8gb 1 billion row file
        key = "1M"

    return files[key], key
