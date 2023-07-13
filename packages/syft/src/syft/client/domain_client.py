# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

# third party
from tqdm import tqdm

# relative
from ..img.base64 import base64read
from ..serde.serializable import serializable
from ..service.dataset.dataset import Contributor
from ..service.dataset.dataset import CreateAsset
from ..service.dataset.dataset import CreateDataset
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.user.roles import Roles
from ..service.user.user_roles import ServiceRole
from ..types.uid import UID
from ..util.fonts import fonts_css
from ..util.util import get_mb_size
from .api import APIModule
from .client import SyftClient
from .client import login

if TYPE_CHECKING:
    # relative
    from ..service.project.project import Project


def add_default_uploader(
    user, obj: Union[CreateDataset, CreateAsset]
) -> Union[CreateDataset, CreateAsset]:
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
        obj.contributors.append(uploader)
    obj.uploader = uploader
    return obj


@serializable()
class DomainClient(SyftClient):
    def __repr__(self) -> str:
        return f"<DomainClient: {self.name}>"

    def upload_dataset(self, dataset: CreateDataset) -> Union[SyftSuccess, SyftError]:
        # relative
        from ..types.twin_object import TwinObject

        user = self.users.get_current_user()
        dataset = add_default_uploader(user, dataset)
        for i in range(len(dataset.assets)):
            asset = dataset.assets[i]
            dataset.assets[i] = add_default_uploader(user, asset)

        dataset._check_asset_must_contain_mock()
        dataset_size = 0

        for asset in tqdm(dataset.asset_list):
            print(f"Uploading: {asset.name}")
            try:
                twin = TwinObject(private_obj=asset.data, mock_obj=asset.mock)
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
        if valid.ok():
            return self.api.services.dataset.add(dataset=dataset)
        else:
            if len(valid.err()) > 0:
                return tuple(valid.err())
            return valid.err()

    def connect_to_gateway(
        self,
        via_client: Optional[SyftClient] = None,
        url: Optional[str] = None,
        port: Optional[int] = None,
        handle: Optional["NodeHandle"] = None,  # noqa: F821
        **kwargs,
    ) -> None:
        if via_client is not None:
            client = via_client
        elif handle is not None:
            client = handle.client
        else:
            client = login(url=url, port=port, **kwargs)
            if isinstance(client, SyftError):
                return client

        res = self.exchange_route(client)
        if isinstance(res, SyftSuccess):
            return SyftSuccess(
                message=f"Connected {self.metadata.node_type} to {client.name} gateway"
            )
        return res

    @property
    def data_subject_registry(self) -> Optional[APIModule]:
        if self.api.has_service("data_subject"):
            return self.api.services.data_subject
        return None

    @property
    def code(self) -> Optional[APIModule]:
        if self.api.has_service("code"):
            return self.api.services.code
        return None

    @property
    def requests(self) -> Optional[APIModule]:
        if self.api.has_service("request"):
            return self.api.services.request
        return None

    @property
    def datasets(self) -> Optional[APIModule]:
        if self.api.has_service("dataset"):
            return self.api.services.dataset
        return None

    @property
    def projects(self) -> Optional[APIModule]:
        if self.api.has_service("project"):
            return self.api.services.project
        return None

    def get_project(
        self,
        name: str = None,
        uid: UID = None,
    ) -> Optional[Project]:
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

        return f"""
        <style>
            {fonts_css}

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
                <!-- <strong>Institution:</strong> TODO<br /> -->
                <!-- <strong>Owner:</strong> TODO<br /> -->
                <strong>URL:</strong> {getattr(self.connection, 'url', '')}<br />
                <!-- <strong>PyGrid Admin:</strong> TODO<br /> -->
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
