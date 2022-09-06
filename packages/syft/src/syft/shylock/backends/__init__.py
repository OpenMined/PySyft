# type: ignore
class ShylockAsyncBackend:
    @staticmethod
    def _check():
        raise NotImplementedError()

    async def acquire(self, name: str, block: bool = True):
        raise NotImplementedError()

    async def release(self, name: str):
        raise NotImplementedError()


class ShylockSyncBackend:
    @staticmethod
    def _check():
        raise NotImplementedError()

    def acquire(self, name: str, block: bool = True):
        raise NotImplementedError()

    def release(self, name: str):
        raise NotImplementedError()
