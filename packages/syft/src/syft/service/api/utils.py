# stdlib
import builtins as __builtin__
import datetime
import sys
from typing import Any

# relative
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..context import AuthedServiceContext
from ..job.job_stash import Job
from ..response import SyftError


def print(
    context: AuthedServiceContext,
    log_id: UID,
    *args: Any,
    sep: str = " ",
    end: str = "\n",
) -> str | None:
    def to_str(arg: Any) -> str:
        if isinstance(arg, bytes):
            return arg.decode("utf-8")
        if isinstance(arg, Job):
            return f"JOB: {arg.id}"
        if isinstance(arg, SyftError):
            return f"JOB: {arg.message}"
        if isinstance(arg, ActionObject):
            return str(arg.syft_action_data)
        return str(arg)

    new_args = [to_str(arg) for arg in args]
    new_str = sep.join(new_args) + end
    if context.server is not None:
        context.server.services.log.append(context=context, uid=log_id, new_str=new_str)
    time = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
    return __builtin__.print(
        f"{time} FUNCTION LOG :",
        *new_args,
        end=end,
        sep=sep,
        file=sys.stderr,
    )
