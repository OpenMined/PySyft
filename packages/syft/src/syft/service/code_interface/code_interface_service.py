# stdlib
# stdlib
from typing import Union

# relative
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .code_interface import CodeInterface
from .code_interface_stash import CodeInterfaceStash


class CodeInterfaceService(AbstractService):
    store: DocumentStore
    stash: CodeInterfaceStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = CodeInterfaceStash(store=store)

    @service_method(path="code_interface.submit_version", name="submit_version")
    def submit_version(
        self, context: AuthedServiceContext, code: Union[SubmitUserCode, UserCode]
    ) -> Union[SyftSuccess, SyftError]:
        user_code_service = context.node.get_service("usercodeservice")

        if isinstance(code, SubmitUserCode):
            result = user_code_service.submit(context=context, code=code)
            if isinstance(result, SyftError):
                return result

            uid = UID.from_string(result.message.split(" ")[-1])
            code = user_code_service.get_by_uid(context=context, uid=uid)

        elif isinstance(code, UserCode):
            result = user_code_service.get_by_uid(context=context, uid=code.id)
            if isinstance(result, SyftError):
                return result
            code = result

        result_code_interface = self.stash.get_by_service_func_name(
            credentials=context.credentials, service_func_name=code.service_func_name
        )
        if result_code_interface.is_err():
            return SyftError(message=result_code_interface.err())

        code_interface = result_code_interface.ok()
        if code_interface is None:
            code_interface = CodeInterface(
                id=UID(),
                node_uid=context.node.id,
                user_verify_key=context.credentials,
                service_func_name=code.service_func_name,
            )

            result = self.stash.set(credentials=context.credentials, obj=code_interface)
            if result.is_err():
                return SyftError(message=result.err())

        code_interface.add_code(code=code)
        result = self.stash.update(credentials=context.credentials, obj=code_interface)
        if result.is_err():
            return SyftError(message=result.err())

        return SyftSuccess(message="Code version submit success")
