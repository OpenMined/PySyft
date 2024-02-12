# stdlib
from io import BytesIO
import math
from queue import Queue
import threading
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
import boto3
from botocore.client import BaseClient as S3BaseClient
from botocore.client import ClientError as BotoClientError
from botocore.client import Config
import requests
from tqdm import tqdm
from typing_extensions import Self

# relative
from . import BlobDeposit
from . import BlobRetrieval
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from ...serde.serializable import serializable
from ...service.blob_storage.remote_profile import AzureRemoteProfile
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...service.service import from_api_or_context
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SeaweedSecureFilePathLocation
from ...types.blob_storage import SecureFilePathLocation
from ...types.grid_url import GridURL
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...util.constants import DEFAULT_TIMEOUT

WRITE_EXPIRATION_TIME = 900  # seconds
DEFAULT_FILE_PART_SIZE = (1024**3) * 5  # 5GB
DEFAULT_UPLOAD_CHUNK_SIZE = 819200


@serializable()
class SeaweedFSBlobDepositV1(BlobDeposit):
    __canonical_name__ = "SeaweedFSBlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    urls: List[GridURL]


@serializable()
class SeaweedFSBlobDeposit(BlobDeposit):
    __canonical_name__ = "SeaweedFSBlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_2

    urls: List[GridURL]
    size: int

    def write(self, data: BytesIO) -> Union[SyftSuccess, SyftError]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )

        etags = []

        try:
            no_lines = 0
            # this loops over the parts, we have multiple parts to allow for
            # concurrent uploads of a single file. (We are currently not using that)
            # a part may for instance be 5GB
            # parts are then splitted into chunks which are MBs (order of magnitude)
            part_size = math.ceil(self.size / len(self.urls))
            chunk_size = DEFAULT_UPLOAD_CHUNK_SIZE

            # this is the total nr of chunks in all parts
            total_iterations = math.ceil(part_size / chunk_size) * len(self.urls)

            with tqdm(
                total=total_iterations,
                desc=f"Uploading progress",  # noqa
            ) as pbar:
                for part_no, url in enumerate(
                    self.urls,
                    start=1,
                ):
                    if api is not None:
                        blob_url = api.connection.to_blob_route(
                            url.url_path, host=url.host_or_ip
                        )
                    else:
                        blob_url = url

                    # read a chunk untill we have read part_size
                    class PartGenerator:
                        def __init__(self):
                            self.no_lines = 0

                        def async_generator(self, chunk_size=DEFAULT_UPLOAD_CHUNK_SIZE):
                            item_queue: Queue = Queue()
                            threading.Thread(
                                target=self.add_chunks_to_queue,
                                kwargs={"queue": item_queue, "chunk_size": chunk_size},
                                daemon=True,
                            ).start()
                            item = item_queue.get()
                            while item != 0:
                                yield item
                                pbar.update(1)
                                item = item_queue.get()

                        def add_chunks_to_queue(
                            self, queue, chunk_size=DEFAULT_UPLOAD_CHUNK_SIZE
                        ):
                            """Creates a data geneator for the part"""
                            n = 0

                            while n * chunk_size <= part_size:
                                try:
                                    chunk = data.read(chunk_size)
                                    self.no_lines += chunk.count(b"\n")
                                    n += 1
                                    queue.put(chunk)
                                except BlockingIOError:
                                    # if end of file, stop
                                    queue.put(0)
                            # if end of part, stop
                            queue.put(0)

                    gen = PartGenerator()

                    response = requests.put(
                        url=str(blob_url),
                        data=gen.async_generator(chunk_size),
                        timeout=DEFAULT_TIMEOUT,
                        stream=True,
                    )

                    response.raise_for_status()
                    no_lines += gen.no_lines
                    etag = response.headers["ETag"]
                    etags.append({"ETag": etag, "PartNumber": part_no})

        except requests.RequestException as e:
            print(e)
            return SyftError(message=str(e))

        mark_write_complete_method = from_api_or_context(
            func_or_path="blob_storage.mark_write_complete",
            syft_node_location=self.syft_node_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        return mark_write_complete_method(
            etags=etags, uid=self.blob_storage_entry_id, no_lines=no_lines
        )


@migrate(SeaweedFSBlobDeposit, SeaweedFSBlobDepositV1)
def downgrade_seaweedblobdeposit_v2_to_v1():
    return [
        drop(["size"]),
    ]


@migrate(SeaweedFSBlobDepositV1, SeaweedFSBlobDeposit)
def upgrade_seaweedblobdeposit_v1_to_v2():
    return [
        make_set_default("size", 1),
    ]


@serializable()
class SeaweedFSClientConfig(BlobStorageClientConfig):
    host: str
    port: int
    mount_port: Optional[int] = None
    access_key: str
    secret_key: str
    region: str
    default_bucket_name: str = "defaultbucket"
    remote_profiles: Dict[str, AzureRemoteProfile] = {}

    @property
    def endpoint_url(self) -> str:
        grid_url = GridURL(host_or_ip=self.host, port=self.port)
        return grid_url.url

    @property
    def mount_url(self) -> str:
        if self.mount_port is None:
            raise ValueError("Seaweed should be configured with a mount port to mount")
        return f"http://{self.host}:{self.mount_port}/configure_azure"


@serializable()
class SeaweedFSClient(BlobStorageClient):
    config: SeaweedFSClientConfig

    def connect(self) -> BlobStorageConnection:
        return SeaweedFSConnection(
            client=boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                config=Config(signature_version="s3v4"),
                region_name=self.config.region,
            ),
            default_bucket_name=self.config.default_bucket_name,
            config=self.config,
        )


@serializable()
class SeaweedFSConnection(BlobStorageConnection):
    client: S3BaseClient
    default_bucket_name: str
    config: SeaweedFSClientConfig

    def __init__(
        self,
        client: S3BaseClient,
        default_bucket_name: str,
        config: SeaweedFSClientConfig,
    ):
        self.client = client
        self.default_bucket_name = default_bucket_name
        self.config = config

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.client.close()

    def read(
        self, fp: SecureFilePathLocation, type_: Optional[Type], bucket_name=None
    ) -> BlobRetrieval:
        if bucket_name is None:
            bucket_name = self.default_bucket_name
        # this will generate the url, the SecureFilePathLocation also handles the logic
        # that decides whether to use a direct connection to azure/aws/gcp or via seaweed
        return fp.generate_url(self, type_, bucket_name)

    def allocate(
        self, obj: CreateBlobStorageEntry
    ) -> Union[SecureFilePathLocation, SyftError]:
        try:
            file_name = obj.file_name
            result = self.client.create_multipart_upload(
                Bucket=self.default_bucket_name,
                Key=file_name,
            )
            upload_id = result["UploadId"]
            return SeaweedSecureFilePathLocation(upload_id=upload_id, path=file_name)
        except BotoClientError as e:
            return SyftError(
                message=f"Failed to allocate space for {obj} with error: {e}"
            )

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        total_parts = math.ceil(obj.file_size / DEFAULT_FILE_PART_SIZE)

        urls = [
            GridURL.from_url(
                self.client.generate_presigned_url(
                    ClientMethod="upload_part",
                    Params={
                        "Bucket": self.default_bucket_name,
                        "Key": obj.location.path,
                        "UploadId": obj.location.upload_id,
                        "PartNumber": i + 1,
                    },
                    ExpiresIn=WRITE_EXPIRATION_TIME,
                )
            )
            for i in range(total_parts)
        ]
        return SeaweedFSBlobDeposit(
            blob_storage_entry_id=obj.id, urls=urls, size=obj.file_size
        )

    def complete_multipart_upload(
        self,
        blob_entry: BlobStorageEntry,
        etags: List,
    ) -> Union[SyftError, SyftSuccess]:
        try:
            self.client.complete_multipart_upload(
                Bucket=self.default_bucket_name,
                Key=blob_entry.location.path,
                MultipartUpload={"Parts": etags},
                UploadId=blob_entry.location.upload_id,
            )
            return SyftSuccess(message="Successfully saved file.")
        except BotoClientError as e:
            return SyftError(message=str(e))

    def delete(
        self,
        fp: SecureFilePathLocation,
    ) -> Union[SyftSuccess, SyftError]:
        try:
            self.client.delete_object(Bucket=self.default_bucket_name, Key=fp.path)
            return SyftSuccess(message="Successfully deleted file.")
        except BotoClientError as e:
            return SyftError(message=str(e))


@serializable()
class SeaweedFSConfig(BlobStorageConfig):
    client_type: Type[BlobStorageClient] = SeaweedFSClient
    client_config: SeaweedFSClientConfig
