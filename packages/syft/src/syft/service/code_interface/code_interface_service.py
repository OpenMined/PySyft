
#stdlib
from typing import Union

from ..service import AbstractService
from ..service import service_method
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..user.user_roles import GUEST_ROLE_LEVEL
from .code_interface import CodeInterfaceStash
from ..code.user_code import UserCode

class CodeInterfaceService(AbstractService):
    store: DocumentStore
    stash: CodeInterfaceStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = CodeInterfaceStash(store=store)

    @service_method(path="code_interface.submit", name="submit_code_version", roles=GUEST_ROLE_LEVEL)
    def submit_code_version(
            self, context: AuthedServiceContext, code:UserCode
            )-> Union[SyftSuccess, SyftError]:
        
        result = self.stash.set(context.credentials, code.to(UserCode, context=context))
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message=f"New Version for or function {code.service_func_name} Submitted Successfully")