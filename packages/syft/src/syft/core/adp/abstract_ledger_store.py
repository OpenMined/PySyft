# future
from __future__ import annotations

# stdlib
from typing import Any

# third party
from nacl.signing import VerifyKey


class AbstractDataSubjectLedger:
    store: AbstractLedgerStore
    user_key: Any
    pass


class AbstractLedgerStore:
    def get(self, key: VerifyKey) -> AbstractDataSubjectLedger:
        raise NotImplementedError

    def set(self, key: VerifyKey, value: AbstractDataSubjectLedger) -> None:
        raise NotImplementedError
