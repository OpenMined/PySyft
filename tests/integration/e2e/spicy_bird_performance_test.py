# stdlib
from datetime import datetime
from pathlib import Path
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from contextlib import contextmanager

# third party
import numpy as np
import pandas as pd
import pytest
import os

# syft absolute
import syft as sy
from syft import Domain
from syft.core.adp.entity import Entity
# from syft.util import download_file
# from syft.util import get_root_data_path
# from syft.util import get_tracer


def str_to_bool(bool_str: Optional[str]) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result

def get_root_data_path() -> Path:
    # get the PySyft / data directory to share datasets between notebooks
    # on Linux and MacOS the directory is: ~/.syft/data"
    # on Windows the directory is: C:/Users/$USER/.syft/data

    data_dir = Path.home() / ".syft" / "data"

    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def download_file(url: str, full_path: Union[str, Path]) -> Path:
    if not os.path.exists(full_path):
        # TODO: Rasswanth fix the SSL Error.
        r = requests.get(url, allow_redirects=True, verify=verify_tls())  # nosec
        path = os.path.dirname(full_path)
        os.makedirs(path, exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(r.content)
    return Path(full_path)


_tracer = None


def get_tracer(service_name: Optional[str] = None) -> Any:
    global _tracer
    if _tracer is not None:  # type: ignore
        return _tracer  # type: ignore

    PROFILE_MODE = str_to_bool(os.environ.get("PROFILE", "False"))
    PROFILE_MODE = False
    if not PROFILE_MODE:

        class NoopTracer:
            @contextmanager
            def start_as_current_span(*args: Any, **kwargs: Any) -> Any:
                yield None

        _tracer = NoopTracer()
        return _tracer
    
    print("Profile mode with OpenTelemetry enabled")
    if service_name is None:
        service_name = os.environ.get("SERVICE_NAME", "client")

    jaeger_host = os.environ.get("JAEGER_HOST", "localhost")
    jaeger_port = int(os.environ.get("JAEGER_PORT", "6831"))

    # third party
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.resources import SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    trace.set_tracer_provider(
        TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    )

    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )

    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))

    _tracer = trace.get_tracer(__name__)
    return _tracer



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
    
    from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor
    
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


DOMAIN1_PORT = 8083
# DOMAIN1_PORT = 8081


@pytest.mark.e2e
def test_benchmark_datasets() -> None:
    # stdlib
    import os

    os.environ["PROFILE"] = "True"
    tracer = get_tracer("test_benchmark_datasets")

    files, ordered_sizes = download_spicy_bird_benchmark()
    domain = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    benchmark_report = {}
    for size_name in ordered_sizes:
        timeout = 999
        unique_key = str(hash(time.time()))
        benchmark_report[size_name] = {}
        df = pd.read_parquet(files[size_name])

        # make smaller
        df = df[0:1000]

        with tracer.start_as_current_span("upload"):
            upload_time = time_upload(
                domain=domain, size_name=size_name, unique_key=unique_key, df=df
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

    print(benchmark_report)
    # assert False


if __name__ == "__main__":
    # stdlib
    import sys

    # third party
    import pytest

    pytest.main(sys.argv)
