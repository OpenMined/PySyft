# future
from __future__ import annotations

# stdlib
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey
from pydantic import BaseSettings
import redis

# syft absolute
import syft as sy

# relative
from .abstract_ledger_store import AbstractDataSubjectLedger
from .abstract_ledger_store import AbstractLedgerStore


class DictLedgerStore(AbstractLedgerStore):
    def __init__(self, settings: BaseSettings) -> None:
        self.kv_store: Dict[VerifyKey, AbstractDataSubjectLedger] = {}

    def get(self, key: VerifyKey) -> AbstractDataSubjectLedger:
        return self.kv_store[key]

    def set(self, key: VerifyKey, value: AbstractDataSubjectLedger) -> None:
        self.kv_store[key] = value


class RedisLedgerStore(AbstractLedgerStore):
    def __init__(self, settings: Optional[BaseSettings] = None) -> None:

        if settings is None:
            raise Exception("RedisStore requires Settings")
        self.settings = settings
        try:
            # TODO: refactor hard coded host and port to configuration
            self.redis: redis.client.Redis = redis.Redis(host="redis", port=6379, db=1)
        except Exception as e:
            print("failed to load redis", e)
            raise e

    def get(self, key: VerifyKey) -> AbstractDataSubjectLedger:
        try:
            key_str = bytes(key).hex()
            buf = self.redis.get(key_str)
            if buf is None:
                raise KeyError()
            return sy.deserialize(buf, from_bytes=True)
        except Exception as e:
            print(f"Failed to get ledger from database. {e}")
            raise e

    def set(self, key: VerifyKey, value: AbstractDataSubjectLedger) -> None:
        try:
            key_str = bytes(key).hex()
            buf = sy.serialize(value, to_bytes=True)
            self.redis.set(key_str, buf)
        except Exception as e:
            print(f"Failed to set ledger to database. {e}")
            raise e
