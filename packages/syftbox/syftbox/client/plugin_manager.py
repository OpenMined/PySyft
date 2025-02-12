from __future__ import annotations

from typing_extensions import Optional

from syftbox.client.base import PluginManagerInterface, SyftBoxContextInterface
from syftbox.client.exceptions import SyftPluginException
from syftbox.client.plugins.apps import AppRunner
from syftbox.client.plugins.sync.manager import SyncManager


class PluginManager(PluginManagerInterface):
    def __init__(
        self,
        context: SyftBoxContextInterface,
        sync_manager: Optional[SyncManager] = None,
        app_runner: Optional[AppRunner] = None,
        **kwargs: dict,
    ) -> None:
        self.__context = context
        self.__sync_manager = sync_manager
        self.__app_runner = app_runner

    @property
    def sync_manager(self) -> SyncManager:
        """the sync manager. lazily initialized"""
        if self.__sync_manager is None:
            try:
                self.__sync_manager = SyncManager(self.__context)
            except Exception as e:
                raise SyftPluginException(f"Failed to initialize sync manager - {e}") from e
        return self.__sync_manager

    @property
    def app_runner(self) -> AppRunner:
        """the app runner. lazily initialized"""
        if self.__app_runner is None:
            try:
                self.__app_runner = AppRunner(self.__context)
            except Exception as e:
                raise SyftPluginException(f"Failed to initialize app runner - {e}") from e
        return self.__app_runner

    def start(self) -> None:
        self.sync_manager.start()
        self.app_runner.start()

    def stop(self) -> None:
        if self.__sync_manager is not None:
            self.__sync_manager.stop()

        if self.__app_runner is not None:
            self.__app_runner.stop()
