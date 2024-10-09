# stdlib
import asyncio
from datetime import datetime
from enum import Enum
from functools import wraps
import logging
from pathlib import Path
import random
import time

LOGS_DIR = Path(__file__).resolve().parents[1] / ".logs"

logging.Formatter.formatTime = (
    lambda self, record, datefmt=None: datetime.fromtimestamp(record.created).isoformat(
        sep="T",
        timespec="microseconds",
    )
)

DEFAULT_FORMATTER = logging.Formatter(
    "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
)
EVENT_FORMATTER = logging.Formatter(
    "%(asctime)s - %(threadName)s - %(message)s",
)


def make_logger(
    name: str,
    instance: str,
    formatter=DEFAULT_FORMATTER,
    level=logging.INFO,
):
    log_file = f"{int(time.time())}_{instance}"
    log_path = Path(LOGS_DIR, log_file, name).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger


class TestFailure(Exception):
    """Custom exception to signal test failures"""

    pass


class BaseEvent(Enum):
    """Base class for events. Subclass this to define your specific events."""

    pass


class EventManager:
    def __init__(self, name: str):
        self.name = name
        self.events = {}
        self.logger = make_logger("events", instance=name, level=logging.INFO)

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
    def __init__(self, name: str, random_wait=None):
        self.name = name
        self.events = EventManager(name)
        self.random_wait = random_wait

        self.logger = make_logger("activity", instance=name, level=logging.INFO)
        self._elogger = make_logger("executions", instance=name, level=logging.DEBUG)

    def unfired_events(self, events: list[BaseEvent]):
        evts = filter(lambda e: not self.events.is_set(e), events)
        evts = [e.name for e in evts]
        return evts

    @staticmethod
    async def blocking_call(func, /, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    @staticmethod
    async def gather(*tasks):
        return await asyncio.gather(*tasks)


class Simulator:
    def __init__(self, name: str):
        self.name = name

    async def start(self, *tasks, check_events=None, random_wait=None, timeout=60):
        context = SimulatorContext(self.name, random_wait)
        results = None

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task(context) for task in tasks]),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            unfired_events = context.unfired_events(check_events)
            if len(unfired_events) == 0:
                # simulator timed out and all events fired
                return results
            if check_events:
                context._elogger.error(f"Timed out. Unfired Events = {unfired_events}")
            raise TestFailure(
                f"simulator timed out after {timeout}s. Please check logs at {LOGS_DIR} for more details."
            )

        if check_events:
            evts = context.unfired_events(check_events)
            if evts:
                raise TestFailure(f"Unfired events: {evts}")

        return results


def sim_entrypoint(func):
    @wraps(func)
    async def wrapper(ctx: SimulatorContext, *args, **kwargs):
        try:
            ctx._elogger.info(f"Started: {func.__name__}")
            result = await func(ctx, *args, **kwargs)
            ctx._elogger.info(f"Completed: {func.__name__}")
            return result
        except Exception:
            ctx._elogger.error(
                f"sim_entrypoint - {func.__name__} - Unhandled exception",
                exc_info=True,
            )
            raise

    return wrapper


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
                    ctx.logger.info(f"Triggering event: {_trigger.name}")

                return result
            except Exception as e:
                ctx._elogger.error(
                    f"sim_activity - {fsig} - Unhandled exception", exc_info=True
                )
                raise TestFailure(e)

        return wrapper

    return decorator
