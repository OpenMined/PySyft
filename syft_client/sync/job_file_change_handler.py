from pydantic import ConfigDict
from syft_client.sync.callback_mixin import BaseModelCallbackMixin


class JobFileChangeHandler(BaseModelCallbackMixin):
    """Responsible for writing files and checking permissions"""

    # allows overwriting methods (handler for testing)
    model_config = ConfigDict(extra="allow")

    def _handle_file_change(self, path: str, content: str):
        """we need this for monkey patching"""
        self.handle_file_change(path, content)

    def handle_file_change(self, path: str, content: str):
        pass
