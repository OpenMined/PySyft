# stdlib
from datetime import datetime
import time
from typing import Any
import uuid

# third party
import numpy as np
from pympler.asizeof import asizeof
import pytest

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.adp.entity_list import EntityList
from syft.core.store.proxy_dataset import ProxyDataset
from syft.util import size_mb

DOMAIN1_PORT = 9082


def size(obj: Any) -> int:
    return asizeof(obj) / (1024 * 1024)  # MBs


def highest() -> int:
    ii32 = np.iinfo(np.int32)
    # 2147483647
    return ii32.max


def make_bounds(data, bound: int) -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly
    generated b/w 0-1"""
    return np.ones_like(data) * bound


# This fails on Windows CI even though the same code works in a Jupyter Notebook
# on the same Windows CI machine.
@pytest.mark.xfail
@pytest.mark.network
def test_large_message_size() -> None:

    # use to enable mitm proxy
    # from syft.grid.connections.http_connection import HTTPConnection
    # HTTPConnection.proxies = {"http": "http://127.0.0.1:8080"}

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    # currently the database wont accept more than around 200mb
    if sy.flags.APACHE_ARROW_TENSOR_SERDE is False:
        ndim = 5500  # 510.76 MB
        # 127 MB in Protobuf
    else:
        ndim = 5500
        # 121 MB in PyArrow

    # rabbitmq recommends max is 512 MB
    # rabbitmq.conf has max_message_size = 536870912
    x = np.random.randint(-highest(), highest(), size=(ndim, ndim))
    x_bytes = sy.serialize(x, to_bytes=True)
    mb_size = size(x_bytes)
    mb = f"{mb_size} MB"

    if mb_size > 510:
        raise Exception(f"Message size: {mb_size} is too big for RabbitMQ.")

    try:
        start_time = time.time()
        print(f"Sending {mb} sized message")
        x_ptr = x.send(domain_client, tags=[f"{x.shape}", mb])
        x_ptr.block_with_timeout(180)
        total_time = time.time() - start_time
        print(f"Took {total_time}")
        data_rate = mb_size / total_time
        print(f"Send transfer rate: {data_rate}")
    except Exception as e:
        total_time = time.time() - start_time
        print(f"Failed to send {x.shape} in {total_time}. {e}")
        raise e

    try:
        start_time = time.time()
        back = x_ptr.get()
        assert (back == x).all()
        total_time = time.time() - start_time
        data_rate = mb_size / total_time
        print(f"Return transfer rate: {data_rate}")
    except Exception as e:
        print(f"Failed to get data back. {e}")
        raise e


@pytest.mark.network
def test_large_blob_upload() -> None:

    # use to enable mitm proxy
    # from syft.grid.connections.http_connection import HTTPConnection
    # HTTPConnection.proxies = {"http": "http://127.0.0.1:8080"}

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    report = {}

    try:
        # multiplier = 1000
        multiplier = 1
        ndim = 1_000_000

        size_name = f"{multiplier}M"
        if multiplier == 1000:
            size_name = "1B"

        report[size_name] = {}

        rows = 1
        cols = 1
        use_blob_storage = True

        # create tensor
        start_time = time.time()
        upper = highest()
        lower = -highest()
        reference_data = np.random.randint(
            lower, upper, size=(multiplier * ndim, rows, cols), dtype=np.int32
        )

        ndept = True
        if not ndept:
            entities = [Entity(name="ϕhishan") * reference_data.shape[0]]
        else:
            one_hot_lookup = np.array(["ϕhishan"])
            entities_indexed = np.zeros(reference_data.shape[0], dtype=np.uint32)
            entities = EntityList(
                one_hot_lookup=one_hot_lookup, entities_indexed=entities_indexed
            )

        tweets_data = sy.Tensor(reference_data).private(
            min_val=0, max_val=30, entities=entities, ndept=ndept
        )

        report[size_name]["tensor_type"] = type(tweets_data.child).__name__
        end_time = time.time()
        report[size_name]["create_tensor_secs"] = end_time - start_time

        # serde for size
        start_time = time.time()
        tweets_data_size = size_mb(sy.serialize(tweets_data, to_bytes=True))
        end_time = time.time()
        report[size_name]["tensor_bytes_size_mb"] = tweets_data_size
        report[size_name]["tensor_serialize_secs"] = end_time - start_time

        # upload dataset
        start_time = time.time()
        unique_tag = str(uuid.uuid4())
        asset_name = f"{size_name}_tweets_{unique_tag}"
        domain_client.load_dataset(
            assets={asset_name: tweets_data},
            name=f"{unique_tag}",
            description=f"{size_name} - {datetime.now()}",
            use_blob_storage=use_blob_storage,
            skip_checks=True,
        )

        end_time = time.time()
        report[size_name]["upload_tensor_secs"] = end_time - start_time

        # get dataset and tensor back
        start_time = time.time()
        dataset = domain_client.datasets[-1]
        asset_ptr = dataset[asset_name]

        # create new tensor from remote Tensor constructor
        new_tensor_ptr = domain_client.syft.core.tensor.tensor.Tensor(child=asset_ptr)
        new_tensor_ptr.block_with_timeout(
            1 * multiplier
        )  # wait for obj upload and proxy obj creation

        # make sure new object is also in blob storage
        new_tensor_proxy = new_tensor_ptr.get(proxy_only=True)
        assert isinstance(new_tensor_proxy, ProxyDataset)

        # pointer addition
        add_res_prt = asset_ptr + asset_ptr
        add_res_prt.block_with_timeout(
            1 * multiplier
        )  # wait for obj upload and proxy obj creation

        # make sure new object is also in blob storage
        add_result_proxy = add_res_prt.get(delete_obj=False, proxy_only=True)
        assert isinstance(add_result_proxy, ProxyDataset)

        # compare result to locally generated result
        add_result = add_res_prt.get(delete_obj=False)
        org_result = tweets_data + tweets_data
        assert org_result == add_result

        # get the proxy object
        result_proxy = asset_ptr.get(delete_obj=False, proxy_only=True)
        assert isinstance(result_proxy, ProxyDataset)
        assert "http" not in result_proxy.url  # no protocol
        assert ":" not in result_proxy.url  # no port
        assert result_proxy.url.startswith("/blob/")  # no host
        assert result_proxy.shape == tweets_data.shape

        # get the real object
        result = asset_ptr.get(delete_obj=False)

        # do we check the client or the GetReprService
        end_time = time.time()
        report[size_name]["download_tensor_secs"] = end_time - start_time

        assert tweets_data == result
    finally:
        print(report)
