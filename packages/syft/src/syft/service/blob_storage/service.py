# stdlib
from pathlib import Path

# third party
import requests

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...service.action.action_object import ActionObject
from ...store.blob_storage import BlobRetrieval
from ...store.blob_storage.on_disk import OnDiskBlobDeposit
from ...store.blob_storage.seaweedfs import SeaweedFSBlobDeposit
from ...store.db.db import DBManager
from ...types.blob_storage import AzureSecureFilePathLocation
from ...types.blob_storage import BlobFileType
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import BlobStorageMetadata
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SeaweedSecureFilePathLocation
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .remote_profile import AzureRemoteProfile
from .remote_profile import RemoteProfileStash
from .stash import BlobStorageStash

BlobDepositType = OnDiskBlobDeposit | SeaweedFSBlobDeposit


@serializable(canonical_name="BlobStorageService", version=1)
class BlobStorageService(AbstractService):
    stash: BlobStorageStash
    remote_profile_stash: RemoteProfileStash

    def __init__(self, store: DBManager) -> None:
        self.stash = BlobStorageStash(store=store)
        self.remote_profile_stash = RemoteProfileStash(store=store)

    @service_method(path="blob_storage.get_all", name="get_all")
    def get_all_blob_storage_entries(
        self, context: AuthedServiceContext
    ) -> list[BlobStorageEntry]:
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(
        path="blob_storage.mount_azure", name="mount_azure", unwrap_on_success=False
    )
    def mount_azure(
        self,
        context: AuthedServiceContext,
        account_name: str,
        account_key: str,
        container_name: str,
        bucket_name: str,
        use_direct_connections: bool = True,
    ) -> SyftSuccess:
        # TODO: fix arguments

        remote_name = f"{account_name}{container_name}"
        remote_name = "".join(ch for ch in remote_name if ch.isalnum())
        args_dict = {
            "account_name": account_name,
            "account_key": account_key,
            "container_name": container_name,
            "remote_name": remote_name,
            "bucket_name": bucket_name,
        }

        new_profile = AzureRemoteProfile(
            profile_name=remote_name,
            account_name=account_name,
            account_key=account_key,
            container_name=container_name,
        )
        remote_profile = self.remote_profile_stash.set(
            context.credentials, new_profile
        ).unwrap()

        seaweed_config = context.server.blob_storage_client.config
        # we cache this here such that we can use it when reading a file from azure
        # from the remote_name
        seaweed_config.remote_profiles[remote_name] = remote_profile

        # TODO: possible wrap this in try catch
        cfg = context.server.blob_store_config.client_config
        init_request = requests.post(url=cfg.mount_url, json=args_dict)  # nosec
        print(init_request.content)
        # TODO check return code
        res = context.server.blob_storage_client.connect().client.list_objects(
            Bucket=bucket_name
        )
        # stdlib
        objects = res["Contents"]
        file_sizes = [object["Size"] for object in objects]
        file_paths = [object["Key"] for object in objects]
        if use_direct_connections:
            secure_file_paths = [
                AzureSecureFilePathLocation(
                    path=file_path,
                    azure_profile_name=remote_name,
                    bucket_name=bucket_name,
                )
                for file_path in file_paths
            ]
        else:
            secure_file_paths = [
                SeaweedSecureFilePathLocation(
                    path=file_path,
                )
                for file_path in file_paths
            ]

        for sfp, file_size in zip(secure_file_paths, file_sizes):
            blob_storage_entry = BlobStorageEntry(
                location=sfp,
                uploaded_by=context.credentials,
                file_size=file_size,
                type_=BlobFileType,
                bucket_name=bucket_name,
            )
            self.stash.set(context.credentials, blob_storage_entry).unwrap()

        return SyftSuccess(message="Mounting Azure Successful!")

    @service_method(
        path="blob_storage.get_files_from_bucket", name="get_files_from_bucket"
    )
    def get_files_from_bucket(
        self, context: AuthedServiceContext, bucket_name: str
    ) -> list:
        bse_list = self.stash.get_all(
            context.credentials, filters={"bucket_name": bucket_name}
        ).unwrap()

        blob_files = []
        for bse in bse_list:
            # self.stash.set(obj=bse, credentials=context.credentials).unwrap()
            # We create an empty ActionObject and set its blob_storage_entry_id to bse.id
            # such that we can call reload_cache which creates
            # the BlobRetrieval (user needs permission to do this)
            # This could be a BlobRetrievalByURL that creates a BlobFile
            # and then sets it in the cache (it does not contain the data, only the BlobFile).
            # In the client, when reading the file, we will creates **another**, blobretrieval
            # object to read the actual data
            blob_file = ActionObject.empty()
            blob_file.syft_blob_storage_entry_id = bse.id
            blob_file.syft_client_verify_key = context.credentials
            if context.server is not None:
                blob_file.syft_server_location = context.server.id
            blob_file.reload_cache()
            blob_files.append(blob_file.syft_action_data)

        return blob_files

    @service_method(path="blob_storage.get_by_uid", name="get_by_uid")
    def get_blob_storage_entry_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> BlobStorageEntry:
        return self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

    @service_method(path="blob_storage.get_metadata", name="get_metadata")
    def get_blob_storage_metadata_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> BlobStorageEntry:
        blob_storage_entry = self.stash.get_by_uid(
            context.credentials, uid=uid
        ).unwrap()
        return blob_storage_entry.to(BlobStorageMetadata)

    # TODO: replace name with `create_blob_retrieval`
    @service_method(
        path="blob_storage.read",
        name="read",
        roles=GUEST_ROLE_LEVEL,
    )
    def read(self, context: AuthedServiceContext, uid: UID) -> BlobRetrieval:
        obj = self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

        with context.server.blob_storage_client.connect() as conn:
            res: BlobRetrieval = conn.read(
                obj.location, obj.type_, bucket_name=obj.bucket_name
            )
            res.syft_blob_storage_entry_id = uid
            res.file_size = obj.file_size
            return res

    @as_result(SyftException)
    def _allocate(
        self,
        context: AuthedServiceContext,
        obj: CreateBlobStorageEntry,
        uploaded_by: SyftVerifyKey | None = None,
    ) -> BlobDepositType:
        """
        Allocate a secure location for the blob storage entry.

        If uploaded_by is None, the credentials of the context will be used.

        Args:
            context (AuthedServiceContext): context
            obj (CreateBlobStorageEntry): create blob parameters
            uploaded_by (SyftVerifyKey | None, optional): Uploader credentials.
                Can be used to upload on behalf of another user, needed for data migrations.
                Defaults to None.

        Returns:
            BlobDepositType: Blob deposit
        """
        upload_credentials = uploaded_by or context.credentials

        with context.server.blob_storage_client.connect() as conn:
            secure_location = conn.allocate(obj)
            blob_storage_entry = BlobStorageEntry(
                id=obj.id,
                location=secure_location,
                type_=obj.type_,
                mimetype=obj.mimetype,
                file_size=obj.file_size,
                uploaded_by=upload_credentials,
            )
            blob_deposit = conn.write(blob_storage_entry)

        self.stash.set(context.credentials, blob_storage_entry).unwrap()
        return blob_deposit

    @service_method(
        path="blob_storage.allocate",
        name="allocate",
        roles=GUEST_ROLE_LEVEL,
    )
    def allocate(
        self, context: AuthedServiceContext, obj: CreateBlobStorageEntry
    ) -> BlobDepositType:
        return self._allocate(context, obj).unwrap()

    @service_method(
        path="blob_storage.allocate_for_user",
        name="allocate_for_user",
        roles=ADMIN_ROLE_LEVEL,
    )
    def allocate_for_user(
        self,
        context: AuthedServiceContext,
        obj: CreateBlobStorageEntry,
        uploaded_by: SyftVerifyKey,
    ) -> BlobDepositType:
        return self._allocate(context, obj, uploaded_by).unwrap()

    @service_method(
        path="blob_storage.write_to_disk",
        name="write_to_disk",
        roles=GUEST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def write_to_disk(
        self, context: AuthedServiceContext, uid: UID, data: bytes
    ) -> SyftSuccess:
        obj = self.stash.get_by_uid(
            credentials=context.credentials,
            uid=uid,
        ).unwrap(
            public_message=f"No blob storage entry exists for uid: {uid}, or you have no permissions to read it"
        )

        try:
            Path(obj.location.path).write_bytes(data)
            return SyftSuccess(message="File successfully saved.")
        except Exception as e:
            raise SyftException(public_message=f"Failed to write object to disk: {e}")

    @service_method(
        path="blob_storage.mark_write_complete",
        name="mark_write_complete",
        roles=GUEST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def mark_write_complete(
        self,
        context: AuthedServiceContext,
        uid: UID,
        etags: list,
        no_lines: int | None = 0,
    ) -> SyftSuccess:
        obj = self.stash.get_by_uid(
            credentials=context.credentials,
            uid=uid,
        ).unwrap(
            public_message=f"No blob storage entry exists for uid: {uid}, or you have no permissions to read it"
        )

        obj.no_lines = no_lines
        self.stash.update(
            credentials=context.credentials,
            obj=obj,
        ).unwrap()

        with context.server.blob_storage_client.connect() as conn:
            result = conn.complete_multipart_upload(obj, etags)

        return result

    @service_method(path="blob_storage.delete", name="delete", unwrap_on_success=False)
    def delete(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        obj = self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

        try:
            with context.server.blob_storage_client.connect() as conn:
                try:
                    conn.delete(obj.location)
                except Exception as e:
                    raise SyftException(
                        public_message=f"Failed to delete blob file with id '{uid}'. Error: {e}"
                    )

            self.stash.delete_by_uid(
                context.credentials, uid, has_permission=True
            ).unwrap()
        except Exception as e:
            raise SyftException(
                public_message=f"Failed to delete blob file with id '{uid}'. Error: {e}"
            )

        return SyftSuccess(
            message=f"Blob storage entry with id '{uid}' deleted successfully."
        )


TYPE_TO_SERVICE[BlobStorageEntry] = BlobStorageEntry
