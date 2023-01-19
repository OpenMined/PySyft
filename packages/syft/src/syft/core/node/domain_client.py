# stdlib
import logging
import sys
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import names
import pandas as pd

# relative
from ... import __version__
from ...logger import traceback_and_raise
from ...util import bcolors
from ...util import print_dynamic_log
from ...util import validate_field
from ..common.message import SyftMessage
from ..common.serde.serialize import _serialize as serialize  # noqa: F401
from ..common.uid import UID
from ..io.address import Address
from ..io.location import Location
from ..io.location.specific import SpecificLocation
from ..io.route import Route
from ..io.virtual import VirtualClientConnection
from ..pointer.pointer import Pointer
from ..store.proxy_dataset import ProxyDataset
from ..tensor.tensor import Tensor
from .abstract.node import AbstractNodeClient
from .common.action.exception_action import ExceptionMessage
from .common.client import Client
from .common.client_manager.association_api import AssociationRequestAPI
from .common.client_manager.dataset_api import DatasetRequestAPI
from .common.client_manager.role_api import RoleRequestAPI
from .common.client_manager.user_api import UserRequestAPI
from .common.client_manager.vpn_api import VPNAPI
from .common.node_service.get_remaining_budget.get_remaining_budget_messages import (
    GetRemainingBudgetMessage,
)
from .common.node_service.node_setup.node_setup_messages import GetSetUpMessage
from .common.node_service.object_request.object_request_messages import (
    CreateBudgetRequestMessage,
)
from .common.node_service.object_transfer.object_transfer_messages import (
    LoadObjectMessage,
)
from .common.node_service.request_receiver.request_receiver_messages import (
    RequestMessage,
)
from .common.node_service.simple.obj_exists import DoesObjectExistMessage
from .common.util import check_send_to_blob_storage
from .common.util import upload_to_s3_using_presigned
from .enums import PyGridClientEnums
from .enums import RequestAPIFields

SAVE_DATASET_TIMEOUT = 300  # seconds


class RequestQueueClient(AbstractNodeClient):
    def __init__(self, client: Client) -> None:
        self.client = client
        self.handlers = RequestHandlerQueueClient(client=client)

        self.users = UserRequestAPI(client=self)
        self.roles = RoleRequestAPI(client=self)
        self.association = AssociationRequestAPI(client=self)
        self.datasets = DatasetRequestAPI(client=self)

    @property
    def requests(self) -> List[RequestMessage]:

        # relative
        from .common.node_service.object_request.object_request_messages import (
            GetAllRequestsMessage,
        )

        msg = GetAllRequestsMessage(
            address=self.client.address, reply_to=self.client.address
        )

        requests = self.client.send_immediate_msg_with_reply(msg=msg).requests  # type: ignore

        for request in requests:
            request.gc_enabled = False
            request.owner_client_if_available = self.client

        return requests

    def get_request_id_from_object_id(self, object_id: UID) -> Optional[UID]:
        for req in self.requests:
            if req.object_id == object_id:
                return req.request_id

        return object_id

    def __getitem__(self, key: Union[str, int]) -> RequestMessage:
        if isinstance(key, str):
            for request in self.requests:
                if key == str(request.id.value):
                    return request
            traceback_and_raise(
                KeyError("No such request found for string id:" + str(key))
            )
        if isinstance(key, int):
            return self.requests[key]
        else:
            traceback_and_raise(KeyError("Please pass in a string or int key"))

        raise Exception("should not get here")

    def __repr__(self) -> str:
        return repr(self.requests)

    def _repr_html_(self) -> str:
        return self.pandas._repr_html_()

    @property
    def pandas(self) -> pd.DataFrame:
        # TODO:
        # Replace all the hardcoded string by enums / abstractions.
        request_lines = [
            {
                "Name": req.user_name,
                "Email": req.user_email,
                "Role": req.user_role,
                "Request Type": req.request_type.upper(),  # type: ignore
                "Status": req.status,
                "Reason": req.request_description,
                "Request ID": req.id,
                "Requested Object's ID": req.object_id
                if req.request_type == "data"
                else None,
                "Requested Object's tags": req.object_tags,
                "Requested Budget": req.requested_budget
                if req.request_type == "budget"
                else None,
                "Current Budget": req.current_budget
                if req.request_type == "budget"
                else None,
            }
            for req in self.requests
        ]
        return pd.DataFrame(request_lines)

    def add_handler(
        self,
        action: str,
        print_local: bool = False,
        log_local: bool = False,
        tags: Optional[List[str]] = None,
        timeout_secs: int = -1,
        element_quota: Optional[int] = None,
    ) -> None:
        handler_opts = self._validate_options(
            id=UID(),
            action=action,
            print_local=print_local,
            log_local=log_local,
            tags=tags,
            timeout_secs=timeout_secs,
            element_quota=element_quota,
        )

        self._update_handler(handler_opts, keep=True)

    def remove_handler(self, key: Union[str, int]) -> None:
        handler_opts = self.handlers[key]

        self._update_handler(handler_opts, keep=False)

    def clear_handlers(self) -> None:
        for handler in self.handlers.handlers:
            id_str = str(handler["id"].value).replace("-", "")
            self.remove_handler(id_str)

    def _validate_options(
        self,
        action: str,
        print_local: bool = False,
        log_local: bool = False,
        tags: Optional[List[str]] = None,
        timeout_secs: int = -1,
        element_quota: Optional[int] = None,
        id: Optional[UID] = None,
    ) -> Dict[str, Any]:
        handler_opts: Dict[str, Any] = {}
        if action not in ["accept", "deny"]:
            traceback_and_raise(Exception("Action must be 'accept' or 'deny'"))
        handler_opts["action"] = action
        handler_opts["print_local"] = bool(print_local)
        handler_opts["log_local"] = bool(log_local)

        handler_opts["tags"] = []
        if tags is not None:
            for tag in tags:
                handler_opts["tags"].append(tag)
        handler_opts["timeout_secs"] = max(-1, int(timeout_secs))
        if element_quota is not None:
            handler_opts["element_quota"] = max(0, int(element_quota))

        if id is None:
            id = UID()
        handler_opts["id"] = id

        return handler_opts

    def _update_handler(self, request_handler: Dict[str, Any], keep: bool) -> None:
        # relative
        from ..common.node_service.request_handler.request_handler_messages import (
            UpdateRequestHandlerMessage,
        )

        msg = UpdateRequestHandlerMessage(
            address=self.client.address, handler=request_handler, keep=keep
        )
        self.client.send_immediate_msg_without_reply(msg=msg)


class RequestHandlerQueueClient:
    def __init__(self, client: Client) -> None:
        self.client = client

    @property
    def handlers(self) -> List[Dict]:
        # relative
        from ..common.node_service.request_handler.request_handler_messages import (
            GetAllRequestHandlersMessage,
        )

        msg = GetAllRequestHandlersMessage(
            address=self.client.address, reply_to=self.client.address
        )
        return validate_field(
            self.client.send_immediate_msg_with_reply(msg=msg), "handlers"
        )

    def __getitem__(self, key: Union[str, int]) -> Dict:
        """
        allow three ways to get an request handler:
            1. use id: str
            2. use tag: str
            3. use index: int
        """
        if isinstance(key, str):
            matches = 0
            match_handler: Optional[Dict] = None
            for handler in self.handlers:
                if key in str(handler["id"].value).replace("-", ""):
                    return handler
                if key in handler["tags"]:
                    matches += 1
                    match_handler = handler
            if matches == 1 and match_handler is not None:
                return match_handler
            elif matches > 1:
                raise KeyError("More than one item with tag:" + str(key))

            raise KeyError("No such request found for string id:" + str(key))
        if isinstance(key, int):
            return self.handlers[key]
        else:
            raise KeyError("Please pass in a string or int key")

    def __repr__(self) -> str:
        return repr(self.handlers)

    @property
    def pandas(self) -> pd.DataFrame:
        def _get_time_remaining(handler: dict) -> int:
            timeout_secs = handler.get("timeout_secs", -1)
            if timeout_secs == -1:
                return -1
            else:
                created_time = handler.get("created_time", 0)
                rem = timeout_secs - (time.time() - created_time)
                return round(rem)

        handler_lines = [
            {
                "tags": handler["tags"],
                "ID": handler["id"],
                "action": handler["action"],
                "remaining time (s):": _get_time_remaining(handler),
            }
            for handler in self.handlers
        ]
        return pd.DataFrame(handler_lines)


class DomainClient(Client):

    domain: SpecificLocation
    requests: RequestQueueClient

    def __init__(
        self,
        name: Optional[str],
        routes: List[Route],
        domain: SpecificLocation,
        network: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        version: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            routes=routes,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
            version=version,
        )

        self.requests = RequestQueueClient(client=self)

        self.post_init()

        self.users = UserRequestAPI(client=self)
        self.roles = RoleRequestAPI(client=self)
        self.association = AssociationRequestAPI(client=self)
        self.datasets = DatasetRequestAPI(client=self)
        self.vpn = VPNAPI(client=self)

    def obj_exists(self, obj_id: UID) -> bool:
        msg = DoesObjectExistMessage(obj_id=obj_id)
        return self.send_immediate_msg_with_reply(msg=msg).payload  # type: ignore

    @property
    def privacy_budget(self) -> float:
        msg = GetRemainingBudgetMessage(address=self.address, reply_to=self.address)
        return self.send_immediate_msg_with_reply(msg=msg).budget  # type: ignore

    def request_budget(
        self,
        eps: float = 0.0,
        reason: str = "",
        skip_checks: bool = False,
    ) -> Any:

        if not skip_checks:
            if eps == 0.0:
                eps = float(input("Please specify how much more epsilon you want:"))

            if reason == "":
                reason = str(
                    input("Why should the domain owner give you more epsilon:")
                )

        msg = CreateBudgetRequestMessage(
            reason=reason,
            budget=eps,
            address=self.address,
        )

        self.send_immediate_msg_without_reply(msg=msg)

        print(
            "Requested "
            + str(eps)
            + " epsilon of budget. Call .privacy_budget to see if your budget has arrived!"
        )

    def load(
        self, obj_ptr: Type[Pointer], address: Address, pointable: bool = False
    ) -> None:
        content = {
            RequestAPIFields.ADDRESS: serialize(address)
            .SerializeToString()  # type: ignore
            .decode(PyGridClientEnums.ENCODING),
            RequestAPIFields.UID: str(obj_ptr.id_at_location.value),
            RequestAPIFields.POINTABLE: pointable,
        }
        self._perform_grid_request(grid_msg=LoadObjectMessage, content=content)

    def setup(self, *, domain_name: Optional[str], **kwargs: Any) -> Any:
        if domain_name is None:
            domain_name = names.get_full_name() + "'s Domain"
            logging.info(
                "No Domain Name provided... picking randomly as: " + domain_name
            )

        kwargs["domain_name"] = domain_name

        response = self.conn.setup(**kwargs)  # type: ignore
        logging.info(response[RequestAPIFields.MESSAGE])

    def reset(self) -> None:
        logging.warning(
            "Node reset will delete the data, as well as the requests. Do you want to continue (y/N)?"
        )
        response = input().lower()
        if response == "y":
            response = self.routes[0].connection.reset()  # type: ignore

    def _perform_grid_request(
        self, grid_msg: Any, content: Optional[Dict[Any, Any]] = None
    ) -> SyftMessage:
        if content is None:
            content = {}
        # Build Syft Message
        content[RequestAPIFields.ADDRESS] = self.address
        content[RequestAPIFields.REPLY_TO] = self.address
        signed_msg = grid_msg(**content).sign(signing_key=self.signing_key)
        # Send to the dest
        response = self.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response

    def get_setup(self, **kwargs: Any) -> Any:
        return self._perform_grid_request(grid_msg=GetSetUpMessage, content=kwargs)

    def apply_to_network(
        self,
        client: Optional[AbstractNodeClient] = None,
        retry: int = 3,
        **metadata: str,
    ) -> None:
        try:
            finish, success = print_dynamic_log("[1/4] Checking Syft Versions")
            if self.version == client.version and client.version == __version__:  # type: ignore
                success.set()
            else:
                print(f"{bcolors.warning('WARNING')}: Syft versions mismatch!")
                print(f"{bcolors.bold('Domain',True)}: {bcolors.underline(self.version,True)}")  # type: ignore
                print(f"{bcolors.bold('Network',True)}: {bcolors.underline(client.version,True)}")  # type: ignore
                print(
                    f"{bcolors.bold('Environment',True)}: {bcolors.underline(__version__,True)}"
                )

                response = input(
                    "\033[1mThis may cause unexpected errors, are you willing to continue? (y/N)\033[0m"
                )
                if response.lower() == "y":
                    success.set()
                else:
                    finish.set()
                    return
            finish.set()

            finish, success = print_dynamic_log("[2/4] Joining Network")
            self.join_network(client=client)
            success.set()
            finish.set()

            timeout = 30
            connected = False
            network_vpn_ip = ""
            domain_vpn_ip = ""

            finish, success = print_dynamic_log("[3/4] Connecting to Secure VPN")
            # get the vpn ips
            while timeout > 0 and connected is False:
                timeout -= 1
                try:
                    vpn_status = self.vpn_status()
                    if vpn_status["connected"]:
                        finish.set()
                        success.set()
                        connected = True
                        continue
                except Exception as e:
                    finish.set()
                    print(f"Failed to get vpn status. {e}")
                time.sleep(1)

            finish, success = print_dynamic_log("[4/4] Registering on the Secure VPN")
            if vpn_status.get("status") == "error":
                raise Exception("Failed to get vpn status.")

            for peer in vpn_status["peers"]:
                # sometimes the hostname we give is different to the one tailscale
                # reports which can convert _ to - so if we change them on both sides
                # we can safely compare
                if peer["hostname"].replace("-", "_") == client.name.replace("-", "_"):  # type: ignore
                    network_vpn_ip = peer["ip"]
            try:
                domain_vpn_ip = vpn_status["host"]["ip"]
            except Exception as e:
                print(f"Failed to get vpn host ip. {e}")

            if network_vpn_ip == "":
                raise Exception(
                    f"Cant find the network node {client.name} in {vpn_status}"  # type: ignore
                )
            if domain_vpn_ip == "":
                raise Exception(f"No host ip in {vpn_status}")

            self.association.create(
                source=domain_vpn_ip,
                target=network_vpn_ip,
                metadata=metadata,
                retry=retry,
            )

            success.set()
            finish.set()
        except Exception as e:
            finish.set()
            print(f"Failed to apply to network with {client}. {e}")

    @property
    def id(self) -> UID:
        return self.domain.id

    @property
    def device(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the Location of that device
        if it is known by the client."""

        return super().device

    @device.setter
    def device(self, new_device: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the Location of that device, this setter
        allows us to save the Location of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        traceback_and_raise(
            Exception("This client points to a domain, you don't need a Device ID.")
        )

    @property
    def vm(self) -> Optional[Location]:
        """This client points to a node, if that node lives within a vm
        or is a vm itself, this property will return the Location of that vm
        if it is known by the client."""

        return super().vm

    @vm.setter
    def vm(self, new_vm: Location) -> Optional[Location]:
        """This client points to a node, if that node lives within a vm
        or is a vm itself and we learn the Location of that vm, this setter
        allows us to save the Location of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        traceback_and_raise(
            Exception("This client points to a device, you don't need a VM Location.")
        )

    def __repr__(self) -> str:
        no_dash = str(self.id).replace("-", "")
        return f"<{type(self).__name__} - {self.name}: {no_dash}>"

    def update_vars(self, state: dict) -> pd.DataFrame:
        for ptr in self.store.store:
            tags = getattr(ptr, "tags", None)
            if tags is not None:
                for tag in tags:
                    state[tag] = ptr
        return self.store.pandas

    def vpn_status(self) -> Dict[str, Any]:
        return self.vpn.get_status()

    def load_dataset(
        self,
        assets: Optional[dict] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        skip_checks: bool = False,
        chunk_size: int = 536870912,  # 500 MB
        use_blob_storage: bool = True,
        **metadata: Dict,
    ) -> None:
        # relative
        from ..tensor.autodp.gamma_tensor import GammaTensor
        from ..tensor.autodp.phi_tensor import PhiTensor

        sys.stdout.write("Loading dataset...")
        if assets is None or not isinstance(assets, dict):
            raise Exception(
                "Missing Assets: Oops!... You forgot to include your data! (or you passed it in the wrong way) \n\n"
                "You must call load_dataset() with a dictionary of assets which are the "
                "private dataset objects (tensors) you wish to allow someone else to study "
                "while PySyft protects it using various privacy enhancing technologies. \n\n"
                "For example, the MNIST dataset is comprised of 6 tensors, so we would create an assets "
                "dictionary with 6 keys (strings) mapping to the 6 tensors of MNIST.\n\n"
                "Please pass in a dictionary where the key is the name of the asset and the value is "
                "the private dataset object (tensor) itself. We recommend uploading assets which "
                "are differential-privacy trackable objects, such as a syft.Tensor() wrapped "
                "numpy.int32 or numpy.float32 object which you "
                "then call .annotate_with_dp_metadata() on. \n\nOnce "
                "you have an assets dictionary call load_dataset(assets=<your dict of objects>)."
            )
        sys.stdout.write("\rLoading dataset... checking assets...")

        if name is None:
            raise Exception(
                "Missing Name: Oops!... You forgot to name your dataset!\n\n"
                "It's important to give your dataset a clear and descriptive name because"
                " the name is the primary way in which potential users of the dataset will"
                " identify it.\n\n"
                'Retry with a string name. I.e., .load_dataset(name="<your name here>)"'
            )
        sys.stdout.write("\rLoading dataset... checking dataset name for uniqueness...")

        # Disabling this for now until we have a more efficient means of querying dataset metadata.
        # TODO: enforce name uniqueness through more efficient means.
        # datasets = self.datasets
        # if not skip_checks:
        #     for i in range(len(datasets)):
        #         d = datasets[i]
        #         sys.stdout.write(".")
        #         if name == d.name:
        #             print(
        #                 "\n\nWARNING - Dataset Name Conflict: A dataset named '"
        #                 + name
        #                 + "' already exists.\n"
        #             )
        #             pref = input("Do you want to upload this dataset anyway? (y/n)")
        #             while pref != "y" and pref != "n":
        #                 pref = input(
        #                     "Invalid input '" + pref + "', please specify 'y' or 'n'."
        #                 )
        #             if pref == "n":
        #                 raise Exception("Dataset loading cancelled.")
        #             else:
        #                 print()  # just for the newline
        #                 break

        sys.stdout.write(
            "\rLoading dataset... checking dataset name for uniqueness..."
            "                                                          "
            "                                                          "
        )

        if description is None:
            raise Exception(
                "Missing Description: Oops!... You forgot to describe your dataset!\n\n"
                "It's *very* important to give your dataset a very clear and complete description"
                " because your users will need to be able to find this dataset (the description is used for search)"
                " AND they will need enough information to be able to know that the dataset is what they're"
                " looking for AND how to use it.\n\n"
                "Start by describing where the dataset came from, how it was collected, and how its formatted."
                "Refer to each object in 'assets' individually so that your users will know which is which. Don't"
                " be afraid to be longwinded. :) Your users will thank you."
            )

        sys.stdout.write(
            "\rLoading dataset... checking asset types...                              "
        )

        if not skip_checks:
            for _, asset in assets.items():

                if not isinstance(asset, Tensor) or not isinstance(
                    getattr(asset, "child", None), (PhiTensor, GammaTensor)
                ):
                    raise Exception(
                        "ERROR: All private assets must have "
                        + "proper Differential Privacy metadata applied.\n"
                        + "\n"
                        + "Example: syft.Tensor([1,2,3,4]).annotate_with_dp_metadata()\n\n"
                        + "and then follow the wizard. 🧙"
                    )
                    # print(
                    #     "\n\nWARNING - Non-DP Asset: You just passed in a asset '"
                    #     + asset_name
                    #     + "' which cannot be tracked with differential privacy because it is a "
                    #     + str(type(asset))
                    #     + " object.\n\n"
                    #     + "This means you'll need to manually approve any requests which "
                    #     + "leverage this data. If this is ok with you, proceed. If you'd like to use "
                    #     + "automatic differential privacy budgeting, please pass in a DP-compatible tensor type "
                    #     + "such as by calling .annotate_with_dp_metadata() "
                    #     + "on a sy.Tensor with a np.int32 or np.float32 inside."
                    # )
                    #
                    # pref = input("Are you sure you want to proceed? (y/n)")
                    #
                    # while pref != "y" and pref != "n":
                    #     pref = input(
                    #         "Invalid input '" + pref + "', please specify 'y' or 'n'."
                    #     )
                    # if pref == "n":
                    #     raise Exception("Dataset loading cancelled.")

        # serialize metadata
        metadata["name"] = bytes(name, "utf-8")  # type: ignore
        metadata["description"] = bytes(description, "utf-8")  # type: ignore

        for k, v in metadata.items():
            if isinstance(v, str):  # type: ignore
                metadata[k] = bytes(v, "utf-8")  # type: ignore

        # blob storage can only be used if domain node has blob storage enabled.
        if not self.settings.get("use_blob_storage", False):
            print(
                "\n\n**Warning**: Blob Storage is disabled on this domain. Switching to database store.\n"
            )
            use_blob_storage = False

        # If one of the assets needs to be send to blob_storage, then store all other
        # assets to blob storage as well
        send_assets_to_blob_storage = any(
            [
                check_send_to_blob_storage(obj=asset, use_blob_storage=use_blob_storage)
                for asset in assets.values()
            ]
        )

        sys.stdout.write("\rLoading dataset... uploading...🚀                        ")

        if send_assets_to_blob_storage:
            # upload to blob storage
            proxy_assets: Dict[str, ProxyDataset] = {}
            # send each asset to blob storage and pack the results back
            for asset_name, asset in assets.items():
                proxy_obj = upload_to_s3_using_presigned(
                    client=self,
                    data=asset,
                    chunk_size=chunk_size,
                    asset_name=asset_name,
                    dataset_name=name,
                )
                proxy_assets[asset_name] = proxy_obj

            dataset_bytes = serialize(proxy_assets, to_bytes=True)
        else:
            # upload directly
            dataset_bytes = serialize(assets, to_bytes=True)

        self.datasets.create_syft(
            dataset=dataset_bytes,
            metadata=metadata,
            platform="syft",
            timeout=SAVE_DATASET_TIMEOUT,
        )
        sys.stdout.write("\rDataset is uploaded successfully !!! 🎉")

        print(
            "\n\nRun `<your client variable>.datasets` to see your new dataset loaded into your machine!"
        )

    def create_user(self, name: str, email: str, password: str, budget: float) -> dict:
        if budget < 0:
            raise ValueError(f"Budget should be a positive number, but got {budget}")
        try:
            self.users.create(name=name, email=email, password=password, budget=budget)
            url = ""
            if not isinstance(self.routes[0].connection, VirtualClientConnection):  # type: ignore
                url = self.routes[0].connection.base_url.host_or_ip  # type: ignore
            response = {"name": name, "email": email, "password": password, "url": url}
            return response
        except Exception as e:
            raise e
