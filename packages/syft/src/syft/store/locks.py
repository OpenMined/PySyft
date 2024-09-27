# third party
from pydantic import BaseModel

# relative
from ..serde.serializable import serializable


@serializable(canonical_name="LockingConfig", version=1)
class LockingConfig(BaseModel):
    lock_name: str = "syft_lock"
    namespace: str | None = None
    expire: int | None = 60
    timeout: int | None = 30
    retry_interval: float = 0.1


@serializable(canonical_name="NoLockingConfig", version=1)
class NoLockingConfig(LockingConfig):
    pass


@serializable(canonical_name="ThreadingLockingConfig", version=1)
class ThreadingLockingConfig(LockingConfig):
    pass
