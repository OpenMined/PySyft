from .util import get_s3_client
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serializable import serializable

@serializable(recursive_serde=True)
class ProxyDataClass:
    __attr_allowlist__ = ["node_name","asset_name","dataset_name", "shape", "dtype"]
    
    def __init__(self,
        asset_name,
        dataset_name,
        shape,
        dtype,
        node_name,
    ):
        self.asset_name = asset_name
        self.dataset_name = dataset_name
        self.shape = shape
        self.dtype = dtype
        self.node_name = node_name

    @property
    def name(self):
        return self.dataset_name + "/" + self.asset_name


    def get_s3_data(self):
        # response = node.datasets.perform_request(
        #     syft_msg=DownloadDataMessage,
        #     context={"filename": self.name}
        # )
        # download_url = response.payload.url
        s3_client = get_s3_client(docker_host=True)
        if s3_client is None:
            return 
        response = s3_client.get_object(Bucket=self.node_name, Key=self.name)
        data = response.get("Body", b"").read()
        # msg = DownloadDataMessage(address=node.address, kwargs={"filename": self.name}, reply_to=node.address)
        # download_url = msg.run(node, verify_key).url
        return deserialize(data, from_bytes=True)