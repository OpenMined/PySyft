# stdlib
import sys
import traceback
from typing import Any
from typing import TYPE_CHECKING

# third party
from IPython.display import display
from result import Err
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..types.base import SyftBaseModel
from ..util.util import sanitize_html

if TYPE_CHECKING:
    # relative
    from .context import AuthedServiceContext


class SyftResponseMessage(SyftBaseModel):
    message: str
    _bool: bool = True
    require_api_update: bool = False

    def is_err(self) -> bool:
        return False

    def is_ok(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        if name in [
            "_bool",
            # "_repr_html_",
            # "message",
            # 'require_api_update',
            # '__bool__',
            # '__eq__',
            # '__repr__',
            # '__str__',
            # '_repr_html_class_',
            # '_repr_html_',
            "_ipython_canary_method_should_not_exist_",
            "_ipython_display_",
            "__canonical_name__",
            "__version__",
        ] or name.startswith("_repr"):
            return super().__getattr__(name)
        display(self)
        raise AttributeError(
            f"You have tried accessing `{name}` on a {type(self).__name__} with message: {self.message}"
        )

    def __bool__(self) -> bool:
        return self._bool

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SyftResponseMessage):
            return (
                self.message == other.message
                and self._bool == other._bool
                and self.require_api_update == other.require_api_update
            )
        return self._bool == other

    def __repr__(self) -> str:
        _class_name_ = type(self).__name__
        return f"{_class_name_}: {self.message}"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def _repr_html_class_(self) -> str:
        return "alert-info"

    def _repr_html_(self) -> str:
        return (
            f'<div class="{self._repr_html_class_}" style="padding:5px;">'
            f"<strong>{type(self).__name__}</strong>: "
            f'<pre class="{self._repr_html_class_}" style="display:inline; font-family:inherit;">'
            f"{sanitize_html(self.message)}</pre></div><br/>"
        )


@serializable(canonical_name="SyftError", version=1)
class SyftError(SyftResponseMessage):
    _bool: bool = False
    tb: str | None = None

    @property
    def _repr_html_class_(self) -> str:
        return "alert-danger"

    def to_result(self) -> Err:
        return Err(value=self.message)

    def __bool__(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def is_ok(self) -> bool:
        return False

    @classmethod
    def from_exception(
        cls,
        context: "AuthedServiceContext",
        exc: Exception,
        include_traceback: bool = False,
    ) -> Self:
        # traceback may contain private information
        # relative
        from ..types.errors import SyftException as NewSyftException

        tb = None
        if isinstance(exc, NewSyftException):
            error_msg = exc.get_message(context)
            print(exc.__traceback__)
            print("".join(traceback.format_exception(exc)))

            print(f"foul {exc.get_tb(context)}")
            if include_traceback:
                tb = exc.get_tb(context)
        else:
            # by default only type
            error_msg = f"Something unexpected happened server side {type(exc)}"
            print(f"Error: {exc}")
            print(traceback.format_exc())
            if include_traceback:
                tb = traceback.format_exc()
                # if they can see the tb, they can also see the exception message
                error_msg = f"Something unexpected happened server side {exc}"
        return cls(message=error_msg, tb=tb)


@serializable(canonical_name="SyftSuccess", version=1)
class SyftSuccess(SyftResponseMessage):
    value: Any | None = None

    def is_err(self) -> bool:
        return False

    def is_ok(self) -> bool:
        return True

    @property
    def _repr_html_class_(self) -> str:
        return "alert-success"

    def unwrap_value(self) -> Any:
        return self.value


@serializable(canonical_name="SyftNotReady", version=1)
class SyftNotReady(SyftError):
    _bool: bool = False

    @property
    def _repr_html_class_(self) -> str:
        return "alert-info"


@serializable(canonical_name="SyftWarning", version=1)
class SyftWarning(SyftResponseMessage):
    @property
    def _repr_html_class_(self) -> str:
        return "alert-warning"


@serializable(canonical_name="SyftInfo", version=1)
class SyftInfo(SyftResponseMessage):
    _bool: bool = False

    @property
    def _repr_html_class_(self) -> str:
        return "alert-info"


@serializable(canonical_name="SyftException", version=1)
class SyftException(Exception):
    traceback: bool = False
    traceback_limit: int = 10

    @property
    def _repr_html_class_(self) -> str:
        return "alert-danger"

    def _repr_html_(self) -> str:
        return (
            f'<div class="{self._repr_html_class_}" style="padding:5px;">'
            + f"<strong>{type(self).__name__}</strong>: {sanitize_html(self.args)}</div><br />"
        )

    @staticmethod
    def format_traceback(etype: Any, evalue: Any, tb: Any, tb_offset: Any) -> str:
        line = "---------------------------------------------------------------------------\n"
        template = ""
        template += line
        template += f"{type(evalue).__name__}\n"
        template += line
        template += f"Exception: {evalue}\n"

        if evalue.traceback:
            template += line
            template += "Traceback:\n"
            tb_lines = "".join(traceback.format_tb(tb, evalue.traceback_limit)) + "\n"
            template += tb_lines
            template += line

        return template


def syft_exception_handler(
    shell: Any, etype: Any, evalue: Any, tb: Any, tb_offset: Any = None
) -> None:
    template = evalue.format_traceback(
        etype=etype, evalue=evalue, tb=tb, tb_offset=tb_offset
    )
    sys.stderr.write(template)


try:
    # third party
    from IPython import get_ipython

    get_ipython().set_custom_exc((SyftException,), syft_exception_handler)  # noqa: F821
except Exception:
    pass  # nosec


@serializable(canonical_name="SyftAttributeError", version=1)
class SyftAttributeError(AttributeError, SyftException):
    pass
