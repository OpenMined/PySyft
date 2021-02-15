# stdlib
import asyncio
import atexit
import os
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Optional

# third party
import nest_asyncio

# syft relative
from ...logger import info
from .environment import is_interactive
from .environment import is_jupyter

# Depending on where syft is imported we may need to get the existing event loop or
# create a new one. In the case of Jupyter there is an existing event loop created
# which we need to get otherwise our events might get blocked. If we import via CLI
# there will be no event loop so we need to create one here and set it or the network
# code will try to create its own and in some instances they wont match

try:
    # stdlib
    from asyncio import get_running_loop  # noqa Python >=3.7
except ImportError:  # pragma: no cover
    # stdlib
    from asyncio.events import _get_running_loop as get_running_loop  # pragma: no cover

# first lets get any existing event loop, this will throw an exception in if there is
# no event loop or will return none in older versions of python
loop = None
try:
    loop = get_running_loop()
except RuntimeError:
    pass

# no event loop found, lets create one so that future asyncio imports use the same loop
if loop is None:
    loop = asyncio.new_event_loop()

# set the event loop so its the same everywhere
asyncio._set_running_loop(loop)


# https://github.com/erdewit/nest_asyncio
# patch the event loop to allow nested event loops
nest_asyncio.apply(loop)


def loop_in_thread(loop: TypeAny) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class EventLoopThread:
    __shared_state: TypeDict[str, TypeAny] = {}

    def __init__(self, loop: TypeAny) -> None:
        if "loop" not in self.__shared_state:
            info("Starting Event Loop")
            self.__shared_state["loop"] = loop
        if "thread" not in self.__shared_state:
            info("Starting Event Loop Thread")
            # daemon=True needed to allow REPL exit() to terminate the thread
            # stdlib
            import threading

            t = threading.Thread(target=loop_in_thread, args=(loop,), daemon=True)
            self.__shared_state["thread"] = t
            t.start()
        self.__dict__ = self.__shared_state

    def shutdown(self) -> None:
        if "loop" in self.__shared_state:
            info("Stopping Event Loop")
            loop.run_until_complete(loop.shutdown_asyncgens())  # type: ignore
            loop.call_soon_threadsafe(loop.stop)  # type: ignore
            del self.__shared_state["loop"]

        if "thread" in self.__shared_state:
            info("Stopping Event Loop Thread")
            t = self.__shared_state["thread"]
            t.join()
            del self.__shared_state["thread"]

    def __del__(self) -> None:
        self.shutdown()


event_loop_thread: Optional[EventLoopThread] = None

# allow threads to be disabled in REPL which will require an event loop to be started
# before running import syft, an example use case is the SCONE container which has
# threads disabled
thread_env = str(os.environ.get("SYFT_USE_EVENT_LOOP_THREAD", "1")).lower()
SYFT_USE_EVENT_LOOP_THREAD = thread_env not in {"0", "false"}

# REPL requires us to create the Thread and Exit handler
if not is_jupyter and is_interactive and SYFT_USE_EVENT_LOOP_THREAD:

    event_loop_thread = EventLoopThread(loop=loop)

    def exit_handler() -> None:
        info("Shutting Down Syft")
        if event_loop_thread is not None:
            event_loop_thread.shutdown()

    atexit.register(exit_handler)

__all__ = ["loop", "event_loop_thread"]
