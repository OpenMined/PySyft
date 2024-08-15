# syft absolute
import syft
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.code.user_code import SubmitUserCode
from syft.service.code.user_code import UserCode
from syft.service.code.user_code_service import UserCodeService
from syft.service.context import AuthedServiceContext
from syft.service.enclave.enclave_service import EnclaveService
from syft.service.response import SyftError
from syft.service.response import SyftSuccess


def get_dummy_syft_function(worker: Worker) -> SubmitUserCode:
    root_client = worker.root_client
    dummy_data = [1, 2, 3]
    data = ActionObject.from_obj(dummy_data)
    action_obj = data.send(root_client)

    @syft.syft_function(
        input_policy=syft.ExactMatch(data=action_obj),
        output_policy=syft.SingleExecutionExactOutput(),
    )
    def simple_function(data):
        return sum(data)

    return simple_function


def test_setup_enclave(worker: Worker):
    service: EnclaveService = worker.get_service("enclaveservice")
    dummy_syft_function = get_dummy_syft_function(worker=worker)
    context = AuthedServiceContext(
        server=worker, credentials=worker.signing_key.verify_key
    )
    result = service.setup_enclave(context, code=dummy_syft_function)
    assert isinstance(result, SyftSuccess)


def test_setup_enclave_with_invalid_context_fails(worker: Worker):
    service: EnclaveService = worker.get_service("enclaveservice")
    dummy_syft_function = get_dummy_syft_function(worker=worker)

    context = AuthedServiceContext(
        server=worker, credentials=worker.signing_key.verify_key
    )
    # Try to setup enclave with no server signing key
    context.server.signing_key = None
    result = service.setup_enclave(context, code=dummy_syft_function)
    assert isinstance(result, SyftError)


def test_setup_enclave_twice_with_same_code_fails(worker: Worker):
    service: EnclaveService = worker.get_service("enclaveservice")
    dummy_submitusercode = get_dummy_syft_function(worker=worker)
    context = AuthedServiceContext(
        server=worker, credentials=worker.signing_key.verify_key
    )

    result1 = service.setup_enclave(context, code=dummy_submitusercode)
    assert isinstance(result1, SyftSuccess)

    # Try to setup enclave with the same code again
    result2 = service.setup_enclave(context, code=dummy_submitusercode)
    assert isinstance(result2, SyftError)

    usercode_service: UserCodeService = worker.get_service("usercodeservice")
    usercodes = usercode_service.get_all(context=context)
    assert len(usercodes) == 1
    usercode = usercodes[0]
    assert isinstance(usercode, UserCode)

    # Also test setting up the Enclave with usercode
    result3 = service.setup_enclave(context, code=usercode)
    assert isinstance(result3, SyftError)
