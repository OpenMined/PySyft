# stdlib
from typing import Any
from typing import Tuple

# third party
from pydantic import BaseSettings

# relative
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serializable import serializable
from ...core.common.uid import UID
from .util import custom_presigned_url
from .util import get_s3_client


@serializable(recursive_serde=True)
class ProxyDataClass:
    __attr_allowlist__ = [
        "node_id",
        "asset_name",
        "dataset_name",
        "shape",
        "dtype",
        "url",
    ]

    def __init__(
        self,
        asset_name: str,
        dataset_name: str,
        shape: Tuple[int, ...],
        dtype: str,
        node_id: UID,
    ) -> None:
        self.asset_name = asset_name
        self.dataset_name = dataset_name
        self.shape = shape
        self.dtype = dtype
        self.node_id = node_id
        self.url = ""

    @property
    def name(self) -> str:
        return self.dataset_name + "/" + self.asset_name

    def get_s3_data(self, settings: BaseSettings) -> Any:
        s3_client = get_s3_client(settings=settings)
        if s3_client is None:
            return
        response = s3_client.get_object(Bucket=self.node_id.no_dash, Key=self.name)
        data = response.get("Body", b"").read()
        return deserialize(data, from_bytes=True)

    def generate_presigned_url(self, settings: BaseSettings) -> None:
        s3_client = get_s3_client(settings=settings)

        download_url = custom_presigned_url(
            s3_client,
            "http://localhost:9082",
            ClientMethod="get_object",
            Params={"Bucket": self.node_id.no_dash, "Key": self.name},
            ExpiresIn=settings.S3_PRESIGNED_TIMEOUT_SECS,
            HttpMethod="GET",
        )
        self.url = download_url
