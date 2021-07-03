# stdlib
import asyncio
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from ....lib.python import String
from ....logger import error
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ..common.managers.association_request_manager import AssociationRequestManager
from ..common.managers.dataset_manager import DatasetManager
from ..common.managers.environment_manager import EnvironmentManager
from ..common.managers.group_manager import GroupManager
from ..common.managers.request_manager import RequestManager
from ..common.managers.role_manager import RoleManager
from ..common.managers.setup_manager import SetupManager
from ..common.managers.user_manager import UserManager
from ..common.node import Node
from ..common.service.association_request import AssociationRequestService
from ..common.service.dataset_service import DatasetManagerService
from ..common.service.group_service import GroupManagerService
from ..common.service.role_service import RoleManagerService
from ..common.service.setup_service import SetUpService
from ..common.service.tensor_service import RegisterTensorService
from ..common.service.user_service import UserManagerService
from ..domain.client import DomainClient
from ..domain.domain import Domain
from ..domain.service import RequestAnswerMessageService
from ..domain.service import RequestMessage
from ..domain.service import RequestService
from .client import NetworkClient


class Network(Node):

    network: SpecificLocation

    child_type = Domain
    client_type = NetworkClient
    child_type_client_type = DomainClient

    def __init__(
        self,
        name: Optional[str],
        network: SpecificLocation = SpecificLocation(),
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
        root_key: Optional[VerifyKey] = None,
        db_path: Optional[str] = None,
        db_engine: Any = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
            db_path=db_path,
            db_engine=db_engine,
        )

        # specific location with name
        self.network = SpecificLocation(name=self.name)
        self.root_key = root_key

        # Database Management Instances
        self.users = UserManager(db_engine)
        self.roles = RoleManager(db_engine)
        self.groups = GroupManager(db_engine)
        self.environments = EnvironmentManager(db_engine)
        self.setup = SetupManager(db_engine)
        self.association_requests = AssociationRequestManager(db_engine)
        self.data_requests = RequestManager(db_engine)
        self.datasets = DatasetManager(db_engine)

        # self.immediate_services_without_reply.append(RequestService)
        # self.immediate_services_without_reply.append(AcceptOrDenyRequestService)
        # self.immediate_services_without_reply.append(UpdateRequestHandlerService)

        # self.immediate_services_with_reply.append(RequestAnswerMessageService)
        # self.immediate_services_with_reply.append(GetAllRequestsService)
        # self.immediate_services_with_reply.append(GetAllRequestHandlersService)

        # Grid Domain Services
        # self.immediate_services_with_reply.append(AssociationRequestService)
        # self.immediate_services_with_reply.append(DomainInfrastructureService)
        self.immediate_services_with_reply.append(SetUpService)
        self.immediate_services_with_reply.append(RegisterTensorService)
        self.immediate_services_with_reply.append(RoleManagerService)
        self.immediate_services_with_reply.append(UserManagerService)
        # self.immediate_services_with_reply.append(DatasetManagerService)
        self.immediate_services_with_reply.append(GroupManagerService)
        # self.immediate_services_with_reply.append(TransferObjectService)
        # self.immediate_services_with_reply.append(RequestService)

        self.requests: List[RequestMessage] = list()
        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()
        self.request_handlers: List[Dict[Union[str, String], Any]] = []
        self.handled_requests: Dict[Any, float] = {}

        self.post_init()

    def loud_print(self):
        # tprint("Grid", "alpha")
        print(
            """                          `-+yy+-`                               
                        .:oydddddhyo:.                           
                     `/yhdddddddddddhys:`                        
                 .`   ./shdddddddddhys/.   ``                    
             `./shhs/.`  .:oydddhyo:.   .:osyo:.`                
          `-oydmmmmmmdy+-`  `-//-`  `-/syhhhhhyyo/-`             
        `+hmmmmmmmmmmddddy+`      `/shhhhhhhhhyyyyys/`           
         `-ohdmmmmmmmddy+-`  `::.  `-/syhhhhhhyyys/-`            
      -o-`   ./yhddhs/.  `./shddhs/.`  .:oyyhyo:.   `./.         
      -ddhs/.`  .::`  `-+ydddddddddhs+-`  `--`   `-+oyy-         
      -dddddhs+-   `/shdddddddddddddhhhyo:`   .:+syyyyy-         
      -ddddddddh.  -hdddddddddddddddhhhhyy-  `syyyyyyyy-         
      -dddddhhhh-  -hhhhhhddddddddhhyyysss-  `ssssysyyy-         
      -hhhhhhhhh-  -hhyyyyyyhhddhyysssssss-  `sssssssss-         
       `-+yhhhhh.  -yyyyyyyyyyysssssssssss-  `ssssso/-`          
       `   ./syy.  -yyyyyyyyyyysssssssssss-  `sso:.   `          
      -y+:`   `-`  -yyyyyyyyssssssssssssss-  `-`   `-/o-         
      -hhhyo/.     `+ssyyssssssssssssssss+`     .:+ssss-         
      -yyyyyyys/`     ./osssssssssssso/.`    `/osssssss-         
      -yyyyyyyyy.  ``    `:+sssoso+:.    ``  `sssssssss-         
      -yyyyyyyys.  -so/.     -//-`    .:os-  `sssssssss-         
      `+syyyssss.  -sssso/-`      `-/ossss-  `ssssssss+`         
         .:ossss.  .ssssssss+`  `+ooooosss-  `sssso:.            
            `-+s.  .ssssssooo.  .oooooooos-  `o/-`               
                   .sssoooooo.  .ooooooooo-                      
                   `-/ooooooo.  `ooooooo/-`                      
                       .:+ooo.  `ooo+:.`                         
                          `-/.  `/-`                             
                                                                 
                                                                 
                                                                 
                                                                 
                                                                 
``````````                 ``````````            ```          `` 
``       ``               ```      ```            `           `` 
``       ``  ```     ```  ``             ``````  ```   ````````` 
``     ````   ``    ```   ``    ``````   ```     ```  ```    ``` 
`````````      ``  ```    ``     `````   ``      ```  ``      `` 
``              `` ``     ``        ``   ``      ```  ``      `` 
``              ````      ````    ````   ``      ```  ```    ``` 
``               ```        ````````     ``      ``    ````````` 
               ```                                               
              ```  
                                |\ |  _ |_      _   _ |
                                | \| (- |_ \)/ (_) |  |(
                                
"""
        )

    @property
    def icon(self) -> str:
        return "🔗"

    @property
    def id(self) -> UID:
        return self.network.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        # this needs to be defensive by checking network_id NOT network.id or it breaks
        try:
            return msg.address.network_id == self.id and msg.address.domain is None
        except Exception as e:
            error(f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}")
            return False
