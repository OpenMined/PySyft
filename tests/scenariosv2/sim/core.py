# stdlib
import asyncio
from datetime import datetime
from enum import Enum
from functools import wraps
import logging
import random
import time

EVENTS_LOG = "sim.events.log"
EXECUTIONS_LOG = "sim.executions.log"
ACTIVITY_LOG = "sim.activity.log"

logging.Formatter.formatTime = (
    lambda self, record, datefmt=None: datetime.fromtimestamp(record.created).isoformat(
        sep="T",
        timespec="microseconds",
    )
)

DEFAULT_FORMATTER = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
)
EVENT_FORMATTER = logging.Formatter(
    "%(asctime)s - %(message)s",
)


class TestFailure(Exception):
    """Custom exception to signal test failures"""

    pass


class BaseEvent(Enum):
    """Base class for events. Subclass this to define your specific events."""

    pass


class EventManager:
    def __init__(self):
        self.events = {}
        self.logger = logging.getLogger("events")
        file_handler = logging.FileHandler(EVENTS_LOG, mode="w")
        file_handler.setFormatter(EVENT_FORMATTER)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    async def wait_for(self, event: BaseEvent):
        if event not in self.events:
            self.events[event] = asyncio.Event()
        await self.events[event].wait()

    def trigger(self, event: BaseEvent):
        if event not in self.events:
            self.events[event] = asyncio.Event()
        self.logger.info(f"Triggered: {event.name}")
        self.events[event].set()

    def is_set(self, event: BaseEvent) -> bool:
        if event not in self.events:
            return False
        return self.events[event].is_set()


class SimulatorContext:
    def __init__(self, random_wait=None):
        self.events = EventManager()
        self.random_wait = random_wait

        self.logger = logging.getLogger("activity")
        file_handler = logging.FileHandler(ACTIVITY_LOG, mode="w")
        file_handler.setFormatter(DEFAULT_FORMATTER)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

        # private logger
        self._elogger = logging.getLogger("executions")
        file_handler = logging.FileHandler(EXECUTIONS_LOG, mode="w")
        file_handler.setFormatter(DEFAULT_FORMATTER)
        self._elogger.addHandler(file_handler)
        self._elogger.setLevel(logging.DEBUG)

    def unfired_events(self, events: list[BaseEvent]):
        evts = filter(lambda e: not self.events.is_set(e), events)
        evts = [e.name for e in evts]
        return evts

    @staticmethod
    async def gather(*tasks):
        return asyncio.gather(*tasks)


class Simulator:
    async def start(self, *tasks, check_events=None, random_wait=None, timeout=60):
        context = SimulatorContext(random_wait)
        results = None

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task(context) for task in tasks]),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            if check_events:
                context._elogger.error(
                    f"Timed out. Unfired Events = {context.unfired_events(check_events)}"
                )
            raise TestFailure(f"simulator timed out after {timeout}s")

        if check_events:
            evts = context.unfired_events(check_events)
            if evts:
                raise TestFailure(f"Unfired events: {evts}")

        return results


def sim_entrypoint():
    def decorator(func):
        @wraps(func)
        async def wrapper(ctx: SimulatorContext, *args, **kwargs):
            try:
                ctx._elogger.info(f"Started: {func.__name__}")
                result = await func(ctx, *args, **kwargs)
                ctx._elogger.info(f"Completed: {func.__name__}")
                return result
            except Exception as e:
                ctx._elogger.error(f"{func.__name__} - {str(e)}")
                raise

        return wrapper

    return decorator


def sim_activity(
    wait_for: BaseEvent | list[BaseEvent] | None = None,
    trigger: BaseEvent | None = None,
):
    def decorator(func):
        @wraps(func)
        async def wrapper(ctx: SimulatorContext, *args, **kwargs):
            fsig = f"{func.__name__}({args}, {kwargs})"

            # ! todo: this isn't working
            _wait_for = kwargs.get("wait_for", wait_for)
            _trigger = kwargs.get("after", trigger)

            if _wait_for:
                ctx._elogger.debug(f"Blocked: for={_wait_for} {fsig}")
                if isinstance(_wait_for, list):
                    await asyncio.gather(*[ctx.events.wait_for(e) for e in _wait_for])
                else:
                    await ctx.events.wait_for(_wait_for)
                ctx._elogger.debug(f"Unblocked: {fsig}")

            wait = 0
            if ctx.random_wait:
                wait = random.uniform(*ctx.random_wait)
                await asyncio.sleep(wait)

            try:
                ctx._elogger.info(f"Started: {fsig} time_wait={wait:.3f}s")
                start = time.time()
                result = await func(ctx, *args, **kwargs)
                total = time.time() - start
                ctx._elogger.info(f"Completed: {fsig} time_taken={total:.3f}s")

                if _trigger:
                    ctx.events.trigger(_trigger)

                return result
            except Exception as e:
                ctx._elogger.error(f"{fsig} - {str(e)}")
                raise

        return wrapper

    return decorator
