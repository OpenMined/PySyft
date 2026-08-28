import logging
from pydantic import BaseModel
from typing import Dict, List, Callable

logger = logging.getLogger(__name__)


class BaseModelCallbackMixin(BaseModel):
    callbacks: Dict[str, List[Callable]] = {}

    def add_callback(self, on: str, callback: Callable):
        if on not in self.callbacks:
            self.callbacks[on] = []
        self.callbacks[on].append(callback)

    def on(self, event: str, callback: Callable) -> None:
        """Readable alias for registering a lifecycle callback."""
        self.add_callback(event, callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Fire every callback registered for `event`, ISOLATED from each other
        and from the emitting core.

        No-op when nothing is registered, so emitting is always safe.
        """
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    "callback for event %r failed (%r) exception: %r",
                    event,
                    callback,
                    e,
                )
