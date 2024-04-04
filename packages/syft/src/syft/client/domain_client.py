# future
from __future__ import annotations

# stdlib
from pathlib import Path
import re
from typing import TYPE_CHECKING
from typing import cast

# third party
from hagrid.orchestra import NodeHandle
from loguru import logger
from tqdm import tqdm

# relative
from ..abstract_node import NodeSideType
from ..img.base64 import base64read
from ..serde.serializable import serializable
from ..service.action.action_object import ActionObject
from ..service.code_history.code_history import CodeHistoriesDict
from ..service.code_history.code_history import UsersCodeHistoriesDict
from ..service.dataset.dataset import Contributor
from ..service.dataset.dataset import CreateAsset
from ..service.dataset.dataset import CreateDataset
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.sync.diff_state import ResolvedSyncState
from ..service.sync.sync_state import SyncState
from ..service.user.roles import Roles
from ..service.user.user import UserView
from ..service.user.user_roles import ServiceRole
from ..types.blob_storage import BlobFile
from ..types.uid import UID
from ..util.fonts import FONT_CSS
from ..util.util import get_mb_size
from ..util.util import prompt_warning_message
from .api import APIModule
from .client import SyftClient
from .client import login
from .client import login_as_guest
from .connection import NodeConnection
from .protocol import SyftProtocol

if TYPE_CHECKING:
    # relative
    from ..service.project.project import Project


def _get_files_from_glob(glob_path: str) -> list[Path]:
    files = Path().glob(glob_path)
    return [f for f in files if f.is_file() and not f.name.startswith(".")]


def _get_files_from_dir(dir: Path, recursive: bool) -> list:
    files = dir.rglob("*") if recursive else dir.iterdir()
    return [f for f in files if not f.name.startswith(".") and f.is_file()]


def _contains_subdir(dir: Path) -> bool:
    for item in dir.iterdir():
        if item.is_dir():
            return True
    return False


def add_default_uploader(
    user: UserView, obj: CreateDataset | CreateAsset
) -> CreateDataset | CreateAsset:
    uploader = None
    for contributor in obj.contributors:
        if contributor.role == str(Roles.UPLOADER):
            uploader = contributor
            break

    if uploader is None:
        uploader = Contributor(
            role=str(Roles.UPLOADER),
            name=user.name,
            email=user.email,
        )
        obj.contributors.add(uploader)

    obj.uploader = uploader
    return obj


@serializable()
class DomainClient(SyftClient):
    def __repr__(self) -> str:
        return f"<DomainClient: {self.name}>"

    def upload_dataset(self, dataset: CreateDataset) -> SyftSuccess | SyftError:
        # relative
        from ..types.twin_object import TwinObject

        if self.users is None:
            return SyftError(f"can't get user service for {self}")

        user = self.users.get_current_user()
        dataset = add_default_uploader(user, dataset)
        for i in range(len(dataset.asset_list)):
            asset = dataset.asset_list[i]
            dataset.asset_list[i] = add_default_uploader(user, asset)

        dataset._check_asset_must_contain_mock()
        dataset_size: float = 0.0

        # TODO: Refactor so that object can also be passed to generate warnings

        self.api.connection = cast(NodeConnection, self.api.connection)

        metadata = self.api.connection.get_node_metadata(self.api.signing_key)

        if (
            metadata.show_warnings
            and metadata.node_side_type == NodeSideType.HIGH_SIDE.value
        ):
            message = (
                "You're approving a request on "
                f"{metadata.node_side_type} side {metadata.node_type} "
                "which may host datasets with private information."
            )
            prompt_warning_message(message=message, confirm=True)

        for asset in tqdm(dataset.asset_list):
            print(f"Uploading: {asset.name}")
            try:
                twin = TwinObject(
                    private_obj=asset.data,
                    mock_obj=asset.mock,
                    syft_node_location=self.id,
                    syft_client_verify_key=self.verify_key,
                )
                twin._save_to_blob_storage()
            except Exception as e:
                return SyftError(message=f"Failed to create twin. {e}")
            response = self.api.services.action.set(twin)
            if isinstance(response, SyftError):
                print(f"Failed to upload asset\n: {asset}")
                return response
            asset.action_id = twin.id
            asset.node_uid = self.id
            dataset_size += get_mb_size(asset.data)
        dataset.mb_size = dataset_size
        valid = dataset.check()
        if isinstance(valid, SyftError):
            return valid
        return self.api.services.dataset.add(dataset=dataset)

    # def get_permissions_for_other_node(
    #     self,
    #     items: list[Union[ActionObject, SyftObject]],
    # ) -> dict:
    #     if len(items) > 0:
    #         if not len({i.syft_node_location for i in items}) == 1 or (
    #             not len({i.syft_client_verify_key for i in items}) == 1
    #         ):
    #             raise ValueError("permissions from different nodes")
    #         item = items[0]
    #         api = APIRegistry.api_for(
    #             item.syft_node_location, item.syft_client_verify_key
    #         )
    #         if api is None:
    #             raise ValueError(
    #                 f"Can't access the api. Please log in to {item.syft_node_location}"
    #             )
    #         return api.services.sync.get_permissions(items)
    #     else:
    #         return {}

    def refresh(self) -> None:
        if self._api and self._api.refresh_api_callback:
            self._api.refresh_api_callback()

    def get_sync_state(self) -> SyncState | SyftError:
        state: SyncState = self.api.services.sync._get_state()
        if isinstance(state, SyftError):
            return state

        for uid, obj in state.objects.items():
            if isinstance(obj, ActionObject):
                obj = obj.refresh_object(resolve_nested=False)
                obj.reload_cache()
                state.objects[uid] = obj
        return state

    def apply_state(self, resolved_state: ResolvedSyncState) -> SyftSuccess | SyftError:
        if len(resolved_state.delete_objs):
            raise NotImplementedError("TODO implement delete")
        items = resolved_state.create_objs + resolved_state.update_objs

        action_objects = [x for x in items if isinstance(x, ActionObject)]

        for action_object in action_objects:
            # NOTE permissions are added separately server side
            action_object._send(self, add_storage_permission=False)

        ignored_batches = resolved_state.ignored_batches

        res = self.api.services.sync.sync_items(
            items,
            resolved_state.new_permissions,
            resolved_state.new_storage_permissions,
            ignored_batches,
            unignored_batches=resolved_state.unignored_batches,
        )
        if isinstance(res, SyftError):
            return res

        self._fetch_api(self.credentials)
        return res

    def upload_files(
        self,
        file_list: BlobFile | list[BlobFile] | str | list[str] | Path | list[Path],
        allow_recursive: bool = False,
        show_files: bool = False,
    ) -> SyftSuccess | SyftError:
        if not file_list:
            return SyftError(message="No files to upload")

        if not isinstance(file_list, list):
            file_list = [file_list]  # type: ignore[assignment]
        file_list = cast(list, file_list)

        expanded_file_list: list[BlobFile | Path] = []

        for file in file_list:
            if isinstance(file, BlobFile):
                expanded_file_list.append(file)
                continue

            path = Path(file)

            if re.search(r"[\*\?\[]", str(path)):
                expanded_file_list.extend(_get_files_from_glob(str(path)))
            elif path.is_dir():
                if not allow_recursive and _contains_subdir(path):
                    res = input(
                        f"Do you want to include all files recursively in {path.absolute()}? [y/n]: "
                    ).lower()
                    print(
                        f'{"Recursively uploading all files" if res == "y" else "Uploading files"} in {path.absolute()}'
                    )
                    allow_recursive = res == "y"
                expanded_file_list.extend(_get_files_from_dir(path, allow_recursive))
            elif path.exists():
                expanded_file_list.append(path)

        if not expanded_file_list:
            return SyftError(message="No files to upload were found")

        print(
            f"Uploading {len(expanded_file_list)} {'file' if len(expanded_file_list) == 1 else 'files'}:"
        )

        if show_files:
            for file in expanded_file_list:
                if isinstance(file, BlobFile):
                    print(file.path or file.file_name)
                else:
                    print(file.absolute())

        try:
            result = []
            for file in expanded_file_list:
                if not isinstance(file, BlobFile):
                    file = BlobFile(path=file, file_name=file.name)
                print("Uploading", file.file_name)
                if not file.uploaded:
                    file.upload_to_blobstorage(self)
                result.append(file)

            return ActionObject.from_obj(result).send(self)
        except Exception as err:
            logger.debug("upload_files: Error creating action_object: {}", err)
            return SyftError(message=f"Failed to upload files: {err}")

    def connect_to_gateway(
        self,
        via_client: SyftClient | None = None,
        url: str | None = None,
        port: int | None = None,
        handle: NodeHandle | None = None,  # noqa: F821
        email: str | None = None,
        password: str | None = None,
        protocol: str | SyftProtocol = SyftProtocol.HTTP,
    ) -> SyftSuccess | SyftError | None:
        if isinstance(protocol, str):
            protocol = SyftProtocol(protocol)

        if via_client is not None:
            client = via_client
        elif handle is not None:
            client = handle.client
        else:
            client = (
                login_as_guest(url=url, port=port)
                if email is None
                else login(url=url, port=port, email=email, password=password)
            )
            if isinstance(client, SyftError):
                return client

        res = self.exchange_route(client, protocol=protocol)
        if isinstance(res, SyftSuccess):
            if self.metadata:
                return SyftSuccess(
                    message=f"Connected {self.metadata.node_type} to {client.name} gateway"
                )
            else:
                return SyftSuccess(message=f"Connected to {client.name} gateway")
        return res

    @property
    def data_subject_registry(self) -> APIModule | None:
        if self.api.has_service("data_subject"):
            return self.api.services.data_subject
        return None

    @property
    def code(self) -> APIModule | None:
        # if self.api.refresh_api_callback is not None:
        #     self.api.refresh_api_callback()
        if self.api.has_service("code"):
            return self.api.services.code
        return None

    @property
    def worker(self) -> APIModule | None:
        if self.api.has_service("worker"):
            return self.api.services.worker
        return None

    @property
    def requests(self) -> APIModule | None:
        if self.api.has_service("request"):
            return self.api.services.request
        return None

    @property
    def datasets(self) -> APIModule | None:
        if self.api.has_service("dataset"):
            return self.api.services.dataset
        return None

    @property
    def projects(self) -> APIModule | None:
        if self.api.has_service("project"):
            return self.api.services.project
        return None

    @property
    def code_history_service(self) -> APIModule | None:
        if self.api is not None and self.api.has_service("code_history"):
            return self.api.services.code_history
        return None

    @property
    def code_history(self) -> CodeHistoriesDict:
        return self.api.services.code_history.get_history()

    @property
    def code_histories(self) -> UsersCodeHistoriesDict:
        return self.api.services.code_history.get_histories()

    @property
    def images(self) -> APIModule | None:
        if self.api.has_service("worker_image"):
            return self.api.services.worker_image
        return None

    @property
    def worker_pools(self) -> APIModule | None:
        if self.api.has_service("worker_pool"):
            return self.api.services.worker_pool
        return None

    @property
    def worker_images(self) -> APIModule | None:
        if self.api.has_service("worker_image"):
            return self.api.services.worker_image
        return None

    @property
    def sync(self) -> APIModule | None:
        if self.api.has_service("sync"):
            return self.api.services.sync
        return None

    @property
    def code_status(self) -> APIModule | None:
        if self.api.has_service("code_status"):
            return self.api.services.code_status
        return None

    @property
    def output(self) -> APIModule | None:
        if self.api.has_service("output"):
            return self.api.services.output
        return None

    def get_project(
        self,
        name: str | None = None,
        uid: UID | None = None,
    ) -> Project | None:
        """Get project by name or UID"""

        if not self.api.has_service("project"):
            return None

        if name:
            return self.api.services.project.get_by_name(name)

        elif uid:
            return self.api.services.project.get_by_uid(uid)

        return self.api.services.project.get_all()

    def _repr_html_(self) -> str:
        guest_commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.login</span> - list projects</li>
        <li>
            <span class='syft-code-block'>&lt;your_client&gt;.code.submit?</span> - display function signature
        </li>"""
        ds_commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>
        <li>
            <span class='syft-code-block'>&lt;your_client&gt;.code.submit?</span> - display function signature
        </li>"""

        do_commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.requests</span> - list requests</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.users</span> - list users</li>
        <li>
            <span class='syft-code-block'>&lt;your_client&gt;.requests.submit?</span> - display function signature
        </li>"""

        # TODO: how to select ds/do commands based on self.__user_role

        if (
            self.user_role.value == ServiceRole.NONE.value
            or self.user_role.value == ServiceRole.GUEST.value
        ):
            commands = guest_commands
        elif (
            self.user_role is not None
            and self.user_role.value == ServiceRole.DATA_SCIENTIST.value
        ):
            commands = ds_commands
        elif (
            self.user_role is not None
            and self.user_role.value >= ServiceRole.DATA_OWNER.value
        ):
            commands = do_commands

        command_list = f"""
        <ul style='padding-left: 1em;'>
            {commands}
        </ul>
        """

        small_grid_symbol_logo = base64read("small-grid-symbol-logo.png")

        url = getattr(self.connection, "url", None)
        node_details = f"<strong>URL:</strong> {url}<br />" if url else ""
        if self.metadata is not None:
            node_details += f"<strong>Node Type:</strong> {self.metadata.node_type.capitalize()}<br />"
            node_side_type = (
                "Low Side"
                if self.metadata.node_side_type == NodeSideType.LOW_SIDE.value
                else "High Side"
            )
            node_details += f"<strong>Node Side Type:</strong> {node_side_type}<br />"
            node_details += (
                f"<strong>Syft Version:</strong> {self.metadata.syft_version}<br />"
            )

        return f"""
        <style>
            {FONT_CSS}

            .syft-container {{
                padding: 5px;
                font-family: 'Open Sans';
            }}
            .syft-alert-info {{
                color: #1F567A;
                background-color: #C2DEF0;
                border-radius: 4px;
                padding: 5px;
                padding: 13px 10px
            }}
            .syft-code-block {{
                background-color: #f7f7f7;
                border: 1px solid #cfcfcf;
                padding: 0px 2px;
            }}
            .syft-space {{
                margin-top: 1em;
            }}
        </style>
        <div class="syft-client syft-container">
            <img src="{small_grid_symbol_logo}" alt="Logo"
            style="width:48px;height:48px;padding:3px;">
            <h2>Welcome to {self.name}</h2>
            <div class="syft-space">
                {node_details}
            </div>
            <div class='syft-alert-info syft-space'>
                &#9432;&nbsp;
                This domain is run by the library PySyft to learn more about how it works visit
                <a href="https://github.com/OpenMined/PySyft">github.com/OpenMined/PySyft</a>.
            </div>
            <h4>Commands to Get Started</h4>
            {command_list}
        </div><br />
        """
