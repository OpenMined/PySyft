# stdlib
from collections.abc import Generator
from io import BytesIO
import logging
import math
from queue import Queue
import threading
from typing import Any

# third party
import boto3
from botocore.client import BaseClient as S3BaseClient
from botocore.client import ClientError as BotoClientError
from botocore.client import Config
from botocore.exceptions import ConnectionError
import requests
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_delay
from tenacity import wait_fixed
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
from ...service.response import SyftSuccess
from ...service.service import from_api_or_context
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SeaweedSecureFilePathLocation
from ...types.blob_storage import SecureFilePathLocation
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.server_url import ServerURL
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.uid import UID
from ...util.constants import DEFAULT_TIMEOUT
from ...util.telemetry import instrument_botocore

MAX_QUEUE_SIZE = 100
WRITE_EXPIRATION_TIME = 900  # seconds
DEFAULT_FILE_PART_SIZE = 1024**3  # 1GB
DEFAULT_UPLOAD_CHUNK_SIZE = 1024 * 800  # 800KB

logger = logging.getLogger(__name__)

instrument_botocore()


@serializable()
class SeaweedFSBlobDeposit(BlobDeposit):
    __canonical_name__ = "SeaweedFSBlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    urls: list[ServerURL]
    size: int
    proxy_server_uid: UID | None = None

    @as_result(SyftException)
    def write(self, data: BytesIO) -> SyftSuccess:
        # relative
        api = self.get_api_wrapped()

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
                colour="green",
            ) as pbar:
                for part_no, url in enumerate(
                    self.urls,
                    start=1,
                ):
                    if api.is_ok() and api.unwrap().connection is not None:
                        api = api.unwrap()
                        if self.proxy_server_uid is None:
                            blob_url = api.connection.to_blob_route(  # type: ignore [union-attr]
                                url.url_path, host=url.host_or_ip
                            )
                        else:
                            blob_url = api.connection.stream_via(  # type: ignore [union-attr]
                                self.proxy_server_uid, url.url_path
                            )
                    else:
                        blob_url = url

                    # read a chunk untill we have read part_size
                    class PartGenerator:
                        def __init__(self) -> None:
                            self.no_lines = 0

                        def async_generator(
                            self, chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE
                        ) -> Generator:
                            item_queue: Queue = Queue(maxsize=MAX_QUEUE_SIZE)
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
                            self,
                            queue: Queue,
                            chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
                        ) -> None:
                            """Creates a data geneator for the part"""
                            n = 0

                            try:
                                while n * chunk_size <= part_size:
                                    chunk = data.read(chunk_size)
                                    if not chunk:
                                        break
                                    self.no_lines += chunk.count(b"\n")
                                    n += 1
                                    queue.put(chunk)
                            except BlockingIOError:
                                pass
                            # if end of file or part, stop
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
            raise SyftException(
                public_message=f"Failed to upload file to SeaweedFS - {e}"
            )

        mark_write_complete_method = from_api_or_context(
            func_or_path="blob_storage.mark_write_complete",
            syft_server_location=self.syft_server_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        if mark_write_complete_method is None:
            raise SyftException(public_message="mark_write_complete_method is None")
        return mark_write_complete_method(
            etags=etags, uid=self.blob_storage_entry_id, no_lines=no_lines
        )


@serializable(canonical_name="SeaweedFSClientConfig", version=1)
class SeaweedFSClientConfig(BlobStorageClientConfig):
    host: str
    port: int
    mount_port: int | None = None
    access_key: str
    secret_key: str
    region: str
    default_bucket_name: str = "defaultbucket"
    remote_profiles: dict[str, AzureRemoteProfile] = {}

    @property
    def endpoint_url(self) -> str:
        server_url = ServerURL(host_or_ip=self.host, port=self.port)
        return server_url.url

    @property
    def mount_url(self) -> str:
        if self.mount_port is None:
            raise ValueError("Seaweed should be configured with a mount port to mount")
        return f"http://{self.host}:{self.mount_port}/configure_azure"


@serializable(canonical_name="SeaweedFSClient", version=1)
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


@serializable(canonical_name="SeaweedFSConnection", version=1)
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

        self._check_connection()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.client.close()

    @retry(
        wait=wait_fixed(5),
        stop=stop_after_delay(60),
        retry=retry_if_exception_type(ConnectionError),
    )
    def _check_connection(self) -> dict:
        return self.client.list_buckets()

    def read(
        self,
        fp: SecureFilePathLocation,
        type_: type | None,
        bucket_name: str | None = None,
    ) -> BlobRetrieval:
        if bucket_name is None:
            bucket_name = self.default_bucket_name
        # this will generate the url, the SecureFilePathLocation also handles the logic
        # that decides whether to use a direct connection to azure/aws/gcp or via seaweed
        return fp.generate_url(self, type_, bucket_name)

    def allocate(self, obj: CreateBlobStorageEntry) -> SecureFilePathLocation:
        try:
            file_name = obj.file_name
            result = self.client.create_multipart_upload(
                Bucket=self.default_bucket_name,
                Key=file_name,
            )
            upload_id = result["UploadId"]
            return SeaweedSecureFilePathLocation(upload_id=upload_id, path=file_name)
        except BotoClientError as e:
            raise SyftException(
                public_message=f"Failed to allocate space for {obj} with error: {e}"
            )

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        total_parts = math.ceil(obj.file_size / DEFAULT_FILE_PART_SIZE)

        urls = [
            ServerURL.from_url(
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
        etags: list,
    ) -> SyftSuccess:
        try:
            self.client.complete_multipart_upload(
                Bucket=self.default_bucket_name,
                Key=blob_entry.location.path,
                MultipartUpload={"Parts": etags},
                UploadId=blob_entry.location.upload_id,
            )
            return SyftSuccess(message="Successfully saved file.")
        except BotoClientError as e:
            raise SyftException(public_message=str(e))

    def delete(
        self,
        fp: SecureFilePathLocation,
    ) -> SyftSuccess:
        try:
            self.client.delete_object(Bucket=self.default_bucket_name, Key=fp.path)
            return SyftSuccess(message="Successfully deleted file.")
        except BotoClientError as e:
            raise SyftException(public_message=str(e))


@serializable(canonical_name="SeaweedFSConfig", version=1)
class SeaweedFSConfig(BlobStorageConfig):
    client_type: type[BlobStorageClient] = SeaweedFSClient
    client_config: SeaweedFSClientConfig
