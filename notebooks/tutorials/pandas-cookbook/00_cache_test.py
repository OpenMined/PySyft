# stdlib
import os


def test_cache_download() -> None:
    # third party
    import pandas as pd

    # syft absolute
    from syft.util.util import PANDAS_DATA
    from syft.util.util import autocache

    encoding = {"bikes.csv": "ISO-8859-1"}

    pandas_csvs = {
        "bikes.csv": 310,
        "311-service-requests.csv": 111069,
        "weather_2012.csv": 8784,
        "popularity-contest": 2898,
    }

    for cache_file, size in pandas_csvs.items():
        f = autocache(f"{PANDAS_DATA}/{cache_file}")
        assert os.path.exists(f)
        enc = encoding.get(cache_file, "utf-8")
        df = pd.read_csv(f, encoding=enc)
        assert len(df) == size
