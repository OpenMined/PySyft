# future
from __future__ import annotations

# stdlib
from collections.abc import Callable
from collections.abc import Sequence
import hashlib
import logging
from typing import Any
import uuid
from uuid import UUID as uuid_type

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable

logger = logging.getLogger(__name__)


@serializable(attrs=["value"], canonical_name="UID", version=1)
class UID:
    """A unique ID for every Syft object.

    This object creates a unique ID for every object in the Syft
    ecosystem. This ID is guaranteed to be unique for the server on
    which it is initialized and is very likely to be unique across
    the whole ecosystem (because it is long and randomly generated).

    Nearly all objects within Syft subclass from this object because
    nearly all objects need to have a unique ID. The only major
    exception a the time of writing is the Client object because it
    just points to another object which itself has an id.

    There is no other way in Syft to create an ID for any object.

    """

    __serde_overrides__: dict[str, Sequence[Callable]] = {
        "value": (lambda x: x.bytes, lambda x: uuid.UUID(bytes=bytes(x)))
    }

    __slots__ = "value"
    value: uuid_type

    def __init__(self, value: Self | uuid_type | str | bytes | None = None):
        """Initializes the internal id using the uuid package.

        This initializes the object. Normal use for this object is
        to initialize the constructor with value==None because you
        want to initialize with a novel ID. The only major exception
        is deserialization, wherein a UID object is created with a
        specific id value.

        :param value: if you want to initialize an object with a specific UID, pass it
                      in here. This is normally only used during deserialization.
        :type value: uuid.uuid4, optional
        :return: returns the initialized object
        :rtype: UID

        .. code-block:: python

            from syft.types.uid import UID
            my_id = UID()
        """
        # checks to make sure you've set a proto_type
        super().__init__()

        # if value is not set - create a novel and unique ID.
        if isinstance(value, str):
            value = uuid.UUID(value, version=4)
        elif isinstance(value, bytes):
            value = uuid.UUID(bytes=value, version=4)
        elif isinstance(value, UID):
            value = value.value

        self.value = uuid.uuid4() if value is None else value

    @staticmethod
    def from_string(value: str) -> UID:
        try:
            return UID(value=uuid.UUID(value))
        except ValueError as e:
            logger.critical(f"Unable to convert {value} to UUID. {e}")
            raise e

    @staticmethod
    def with_seed(value: str) -> UID:
        md5 = hashlib.md5(value.encode("utf-8"), usedforsecurity=False)
        return UID(md5.hexdigest())

    def to_string(self) -> str:
        return self.no_dash

    def __str__(self) -> str:
        return self.no_dash

    def __hash__(self) -> int:
        """Hashes the UID for use in dictionaries and sets

        A very common use of UID objects is as a key in a dictionary
        or database. The object must be able to be hashed in order to
        be used in this way. We take the 128-bit int representation of the
        value.

        :return: returns a hash of the object
        :rtype: int

        .. note::
            Note that this probably gets further hashed into a shorter
            representation for most python data-structures.

        .. note::
            Note that we assume that any collisions will be very rare and
            detected by the ObjectStore class in Syft.
        """

        return self.value.int

    def __eq__(self, other: Any) -> bool:
        """Checks to see if two UIDs are the same using the internal object

        This checks to see whether this UID is equal to another UID by
        comparing whether they have the same .value objects. These objects
        come with their own __eq__ function which we assume to be correct.

        :param other: this is the other ID to be compared with
        :type other: Any (note this must be Any or __eq__ fails on other types)
        :return: returns True/False based on whether the objects are the same
        :rtype: bool
        """

        try:
            return self.value == other.value
        except Exception:
            return False

    def __lt__(self, other: Any) -> bool:
        try:
            return self.value < other.value
        except Exception:
            return False

    @staticmethod
    def is_valid_uuid(value: Any) -> bool:
        try:
            UID(value=uuid.UUID(value))
            return True
        except Exception:
            return False

    @property
    def no_dash(self) -> str:
        return str(self.value).replace("-", "")

    @property
    def hex(self) -> str:
        return self.value.hex

    def __repr__(self) -> str:
        """Returns a human-readable version of the ID

        Return a human-readable representation of the UID with brackets
        so that it can be easily spotted when nested inside of the human-
        readable representations of other objects."""

        return f"<{type(self).__name__}: {self.no_dash}>"

    def char_emoji(self, hex_chars: str) -> str:
        base = ord("\U0001f642")
        hex_base = ord("0")
        code = 0
        for char in hex_chars:
            offset = ord(char)
            code += offset - hex_base
        return chr(base + code)

    def string_emoji(self, string: str, length: int, chunk: int) -> str:
        output = []
        part = string[-length:]
        while len(part) > 0:
            part, end = part[:-chunk], part[-chunk:]
            output.append(self.char_emoji(hex_chars=end))
        return "".join(output)

    def emoji(self) -> str:
        return f"<UID:{self.string_emoji(string=str(self.value), length=8, chunk=4)}>"

    def short(self) -> str:
        """Returns a SHORT human-readable version of the ID

        Return a SHORT human-readable version of the ID which
        makes it print nicer when embedded (often alongside other
        UID objects) within other object __repr__ methods."""

        return str(self.value)[:8]

    @property
    def id(self) -> UID:
        return self

    @classmethod
    def _check_or_convert(cls, value: str | uuid.UUID | UID) -> UID:
        if isinstance(value, uuid.UUID):
            return UID(value)
        elif isinstance(value, str):
            return UID.from_string(value)
        elif isinstance(value, cls):
            return value
        else:
            # Ask @Madhava , can we check for  invalid types , even though type annotation is specified.
            return ValueError(  # type: ignore
                f"Incorrect value,type:{value,type(value)} for conversion to UID, expected str | uuid.UUID | Self"
            )


@serializable(attrs=["syft_history_hash"], canonical_name="LineageID", version=1)
class LineageID(UID):
    """Extended UID containing a history hash as well, which is used for comparisons."""

    syft_history_hash: int

    def __init__(
        self,
        value: Self | UID | uuid_type | str | bytes | None = None,
        syft_history_hash: int | None = None,
    ):
        if isinstance(value, LineageID):
            syft_history_hash = value.syft_history_hash
            value = value.value

        super().__init__(value)

        if syft_history_hash is None:
            syft_history_hash = hash(self.value)
        self.syft_history_hash = syft_history_hash

    @property
    def id(self) -> UID:
        return UID(self.value)

    def __hash__(self) -> int:
        return hash((self.syft_history_hash, self.value))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, LineageID):
            return (
                self.id == other.id
                and self.syft_history_hash == other.syft_history_hash
            )
        elif isinstance(other, UID):
            return hash(self) == hash(other)
        else:
            raise ValueError(f"Unsupported comparison: LineageID with {type(other)}")

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.no_dash} - {self.syft_history_hash}>"
