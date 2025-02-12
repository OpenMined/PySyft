from __future__ import annotations

from loguru import logger
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from typing_extensions import Callable, List
from watchdog.events import FileSystemEvent, FileSystemEventHandler

__all__ = ["RpcRequestHandler", "AnyPatternHandler"]


class PatternMatchingHandler(FileSystemEventHandler):
    def __init__(self, patterns: List[str], ignore_directory: bool = True):
        self.spec = PathSpec.from_lines(GitWildMatchPattern, patterns)
        self.patterns = patterns
        self.ignore_directory = ignore_directory

    def dispatch(self, event: FileSystemEvent) -> None:
        if self.ignore_directory and event.is_directory:
            return
        if self.spec.match_file(event.src_path):
            super().dispatch(event)


class RpcRequestHandler(PatternMatchingHandler):
    def __init__(self, handler: Callable[[FileSystemEvent], None]):
        super().__init__(patterns=["**/*.request"])
        self.handler = handler

    def on_any_event(self, event: FileSystemEvent):
        logger.debug(f"FSEvent - {event.event_type} - {event.src_path}")
        self.handler(event)


class AnyPatternHandler(PatternMatchingHandler):
    def __init__(self, patterns: List[str], handler: Callable[[FileSystemEvent], None]):
        super().__init__(patterns)
        self.handler = handler

    def on_any_event(self, event: FileSystemEvent):
        self.handler(event)
