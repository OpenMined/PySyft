# stdlib
from datetime import datetime
import time
import uuid

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.store.proxy_dataset import ProxyDataset
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE
from syft.util import size_mb

DOMAIN1_PORT = 9082


def highest() -> int:
    ii64 = np.iinfo(DEFAULT_INT_NUMPY_TYPE)
    return ii64.max


@pytest.mark.e2e
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
        ndim = 1_000

        size_name = f"{multiplier}K"
        if multiplier == 1000:
            size_name = "1M"

        report[size_name] = {}

        rows = 1
        cols = 1
        use_blob_storage = True

        # create tensor
        start_time = time.time()
        upper = highest()
        lower = -highest()
        reference_data = np.random.randint(
            lower,
            upper,
            size=(multiplier * ndim, rows, cols),
            dtype=DEFAULT_INT_NUMPY_TYPE,
        )

        data_subject_name = "ϕhishan"
        data_subjects = np.broadcast_to(
            np.array(DataSubjectArray([data_subject_name])), reference_data.shape
        )

        tweets_data = sy.Tensor(reference_data).private(
            min_val=0, max_val=30, data_subjects=data_subjects
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
            90 * multiplier
        )  # wait for obj upload and proxy obj creation

        # make sure new object is also in blob storage
        new_tensor_proxy = new_tensor_ptr.get(proxy_only=True)
        assert isinstance(new_tensor_proxy, ProxyDataset)

        # pointer addition
        add_res_prt = asset_ptr + asset_ptr
        add_res_prt.block_with_timeout(
            90 * multiplier
        )  # wait for obj upload and proxy obj creation

        # make sure new object is also in blob storage
        add_result_proxy = add_res_prt.get(delete_obj=False, proxy_only=True)
        assert isinstance(add_result_proxy, ProxyDataset)

        # compare result to locally generated result
        add_result = add_res_prt.get(delete_obj=False)
        org_result = tweets_data + tweets_data

        assert org_result.child == add_result.child

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
