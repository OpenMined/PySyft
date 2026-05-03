# stdlib
from unittest.mock import Mock

# syft absolute
from syft.service.code.user_code_service import HasCodePermissionEnum
from syft.service.code.user_code_service import IsExecutionAllowedEnum
from syft.service.code.user_code_service import UserCodeService


class _Result:
    def __init__(self, value):
        self.value = value

    def unwrap(self):
        return self.value


def _approved_code():
    status = Mock()
    status.get_is_approved.return_value = True

    code = Mock()
    code.get_status.return_value = _Result(status)
    code.is_output_policy_approved.return_value = True
    return code


def _service_with_code_permission():
    service = UserCodeService.__new__(UserCodeService)
    service.has_code_permission = Mock(return_value=HasCodePermissionEnum.ACCEPTED)
    return service


def test_is_execution_allowed_rejects_false_output_policy() -> None:
    service = _service_with_code_permission()
    output_policy = Mock()
    output_policy.is_valid.return_value = False

    result = service.is_execution_allowed(
        code=_approved_code(),
        context=Mock(),
        output_policy=output_policy,
    )

    assert result is IsExecutionAllowedEnum.INVALID_OUTPUT_POLICY


def test_is_execution_allowed_accepts_true_output_policy() -> None:
    service = _service_with_code_permission()
    output_policy = Mock()
    output_policy.is_valid.return_value = True

    result = service.is_execution_allowed(
        code=_approved_code(),
        context=Mock(),
        output_policy=output_policy,
    )

    assert result is IsExecutionAllowedEnum.ALLOWED
