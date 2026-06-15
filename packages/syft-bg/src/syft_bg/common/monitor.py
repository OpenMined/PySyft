"""Base monitor class for polling-based services."""

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional


class Monitor(ABC):
    """Abstract base class for polling monitors.

    Subclasses must implement _check_all_entities() to perform the actual work.
    """

    def __init__(self):
        self._stop_event: Optional[threading.Event] = None

    @abstractmethod
    def _check_all_entities(self):
        """Check all entities and process any that need action."""
        pass

    def run_once(self, interval: Optional[int] = None, duration: Optional[int] = None):
        """Run checks, optionally in a loop with interval and duration."""
        if interval is None:
            self._check_all_entities()
            return

        start_time = time.time()
        print(f"[{self.__class__.__name__}] Started (interval: {interval}s)")

        try:
            while True:
                self._check_all_entities()

                if duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        print(f"[{self.__class__.__name__}] Completed ({elapsed:.0f}s)")
                        break

                time.sleep(interval)

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\n[{self.__class__.__name__}] Stopped ({elapsed:.0f}s)")

    def start(self, interval: int = 10) -> threading.Thread:
        """Start monitoring in a background thread."""
        self._stop_event = threading.Event()

        def run():
            print(f"[{self.__class__.__name__}] Started (interval: {interval}s)")
            while not self._stop_event.is_set():
                try:
                    self._check_all_entities()
                except Exception as e:
                    print(f"[{self.__class__.__name__}] Error: {e}")
                self._stop_event.wait(interval)
            print(f"[{self.__class__.__name__}] Stopped")

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Stop the background monitoring thread."""
        if self._stop_event:
            self._stop_event.set()
