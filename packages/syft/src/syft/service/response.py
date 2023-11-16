# stdlib
import sys
import traceback
from typing import Any

# third party
from result import Err

# relative
from ..serde.serializable import serializable
from ..types.base import SyftBaseModel


class SyftResponseMessage(SyftBaseModel):
    message: str
    _bool: bool = True

    def __bool__(self) -> bool:
        return self._bool

    def __eq__(self, other) -> bool:
        if isinstance(other, SyftResponseMessage):
            return self.message == other.message and self._bool == other._bool
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
            + f"<strong>{type(self).__name__}</strong>: {self.message}</div><br />"
        )


@serializable()
class SyftError(SyftResponseMessage):
    _bool: bool = False

    @property
    def _repr_html_class_(self) -> str:
        return "alert-danger"

    def to_result(self):
        return Err(value=self.message)


@serializable()
class SyftSuccess(SyftResponseMessage):
    @property
    def _repr_html_class_(self) -> str:
        return "alert-success"


@serializable()
class SyftNotReady(SyftResponseMessage):
    _bool: bool = False

    @property
    def _repr_html_class_(self) -> str:
        return "alert-info"


@serializable()
class SyftWarning(SyftResponseMessage):
    @property
    def _repr_html_class_(self) -> str:
        return "alert-warning"


@serializable()
class SyftInfo(SyftResponseMessage):
    _bool: bool = False

    @property
    def _repr_html_class_(self) -> str:
        return "alert-info"


@serializable()
class SyftException(Exception):
    traceback: bool = False
    traceback_limit: int = 10

    @property
    def _repr_html_class_(self) -> str:
        return "alert-danger"

    def _repr_html_(self) -> str:
        return (
            f'<div class="{self._repr_html_class_}" style="padding:5px;">'
            + f"<strong>{type(self).__name__}</strong>: {self.args}</div><br />"
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
    get_ipython().set_custom_exc((SyftException,), syft_exception_handler)  # noqa: F821
except Exception:
    pass  # nosec


@serializable()
class SyftAttributeError(AttributeError, SyftException):
    pass
