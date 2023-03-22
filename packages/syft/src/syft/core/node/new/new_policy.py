from typing import Dict
from typing import Any
from typing import Optional
from enum import Enum
from typing import Union
from typing import Type
from RestrictedPython import compile_restricted
from result import Ok
from result import Result
from typing import List
from .credentials import SyftVerifyKey
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
import inspect
from .serializable import serializable
from .uid import UID
from .response import SyftError
from .response import SyftSuccess
from .datetime import DateTime
from .context import NodeServiceContext
from .context import AuthedServiceContext
from .policy import allowed_ids_only, retrieve_from_db
from .api import NodeView

PyCodeObject = Any

@serializable(recursive_serde=True)
class OutputHistory(SyftObject):
    # version
    __canonical_name__ = "OutputHistory"
    __version__ = SYFT_OBJECT_VERSION_1

    output_time: DateTime
    outputs: Optional[Union[List[UID], Dict[str, UID]]]
    executing_user_verify_key: SyftVerifyKey

@serializable(recursive_serde=True)
class UserPolicyStatus(Enum):
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPROVED = "approved"

class Policy:
    id: UID
    @property
    def policy_code(self) -> str:
        cls = type(self)
        op_code = inspect.getsource(cls)
        return op_code
    
    def public_state() -> None:
        raise NotImplementedError
    
class InputPolicy(Policy):
    inputs: Dict[NodeView, Any]
    node_uid: Optional[UID]
    
    def filter_kwargs() -> None:
        raise NotImplementedError


@serializable(recursive_serde=True)
class ExactMatch(InputPolicy, SyftObject):
    # version
    __canonical_name__ = "ExactMatch"
    __version__ = SYFT_OBJECT_VERSION_1

    def filter_kwargs(
        self, kwargs: Dict[str, Any], context: AuthedServiceContext, code_item_id: UID
    ) -> Dict[str, Any]:
        allowed_inputs = allowed_ids_only(
            allowed_inputs=self.inputs, kwargs=kwargs, context=context
        )
        return retrieve_from_db(
            code_item_id=code_item_id, allowed_inputs=allowed_inputs, context=context
        )

class OutputPolicy(Policy):
    
    output_history: List[OutputHistory] = []
    outputs: List[str] = []
    node_uid: Optional[UID]

    def apply_output(
        self, 
        context: NodeServiceContext,
        outputs: Union[UID, List[UID], Dict[str, UID]],
    ) -> None:
        if isinstance(outputs, UID):
            outputs = [outputs]
        history = OutputHistory(
            output_time=DateTime.now(),
            outputs=outputs,
            executing_user_verify_key=context.credentials,
        )
        self.output_history.append(history)

@serializable(recursive_serde=True)
class OutputPolicyExecuteCount(OutputPolicy, SyftObject):
    __canonical_name__ = "OutputPolicyExecuteCount"
    __version__ = SYFT_OBJECT_VERSION_1

    count: int = 0
    limit: int
    
    def apply_output(
        self, 
        context: NodeServiceContext, 
        outputs: Union[UID, List[UID], Dict[str, UID]]
    ) -> Union[UID, List[UID], Dict[str, UID]]:
        if self.count < self.limit:
            super().apply_output(context, outputs)
            self.count +=1
            return SyftSuccess()
        else:
            return  SyftError(
                message=f"Policy is no longer valid. count: {self.count} >= limit: {self.limit}"
            )
            
    def public_state(self) -> None:
        return {"limit": self.limit, "count": self.count}
            
@serializable(recursive_serde=True)
class OutputPolicyExecuteOnce(OutputPolicyExecuteCount):
    __canonical_name__ = "OutputPolicyExecuteOnce"
    __version__ = SYFT_OBJECT_VERSION_1

    limit: int = 1


class CustomPolicy(Policy):
    init_args: Dict[str, Any] = {}
    init_kwargs: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    
class CustomInputPolicy(CustomPolicy, InputPolicy):
    pass
    
class CustomOutputPolicy(CustomPolicy, OutputPolicy):
    pass


 
@serializable(recursive_serde=True)
class UserPolicy(Policy):
    __canonical_name__ = "UserPolicy"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: Optional[UID]
    user_verify_key: SyftVerifyKey
    raw_code: str
    parsed_code: str
    signature: inspect.Signature
    class_name: str
    unique_name: str
    code_hash: str
    byte_code: PyCodeObject
    status: UserPolicyStatus = UserPolicyStatus.SUBMITTED
    state_type: Optional[Type] = None

    @property
    def byte_code(self) -> Optional[PyCodeObject]:
        return compile_byte_code(self.parsed_code)

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        return SyftSuccess(message="Policy is valid.")
    
    @property
    def policy_code(self) -> str:
        return self.raw_code

def compile_byte_code(parsed_code: str) -> Optional[PyCodeObject]:
    try:
        return compile(parsed_code, "<string>", "exec")
    except Exception as e:
        print("WARNING: to compile byte code", e)
    return None
