# stdlib
import threading
from typing import Any
from typing import Optional

# third party
from typing_extensions import Self


class RecursiveSerdeContext:
    """Context manager singleton that tracks recursive SerDe calls in a thread-safe way"""

    _instance: Optional[Self] = None
    _store = threading.local()

    def __new__(cls) -> Any:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def in_recursion(self) -> bool:
        return self._store.recurse_step > 1

    @property
    def recurse_step(self) -> int:
        return getattr(self._store, "recurse_step", 0)

    def get(self, key: str) -> Any:
        return getattr(self._store, key)

    def set(self, key: str, value: Any) -> None:
        setattr(self._store, key, value)

    def __track_call(self) -> None:
        if not hasattr(self._store, "recurse_step"):
            self._store.recurse_step = 0

        self._store.recurse_step += 1

    def __enter__(self) -> Self:
        self.__track_call()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._store.recurse_step -= 1

        # Clear store when the recursion ends
        if self._store.recurse_step == 0:
            self._store.__dict__.clear()


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
    for_hashing: bool = False,
    debug: bool = False,
) -> Any:
    # relative
    from .recursive import rs_object2proto

    with RecursiveSerdeContext() as ctx:
        if ctx.in_recursion:
            # select start args will be passed to recursive serde invocations
            start_args = ctx.get("start_args")
            for_hashing = start_args.get("for_hashing")
            debug = start_args.get("debug")
        else:
            # save the start args in serde context
            start_args = dict(for_hashing=for_hashing, debug=debug)
            ctx.set("start_args", start_args)

        if debug:
            print(
                f'{"-" * ctx.recurse_step}>',
                f"Obj = {obj.__class__.__name__}@{hex(id(obj))} | Value = {obj}",
            )

        proto = rs_object2proto(obj, for_hashing=for_hashing)

        if to_bytes:
            return proto.to_bytes()

        if to_proto:
            return proto
