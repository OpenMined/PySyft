# stdlib
import logging

# third party
from sqlalchemy.exc import DatabaseError
from typing_extensions import Self

# relative
from ..document_store_errors import StashException

logger = logging.getLogger(__name__)


class StashDBException(StashException):
    """
    See https://docs.sqlalchemy.org/en/20/errors.html#databaseerror

    StashDBException converts a SQLAlchemy DatabaseError into a StashException,
    DatabaseErrors are errors thrown by the database itself, for example when a
    query fails because a table is missing.
    """

    public_message = "There was an error retrieving data. Contact your admin."

    @classmethod
    def from_sqlalchemy_error(cls, e: DatabaseError) -> Self:
        logger.exception(e)

        error_type = e.__class__.__name__
        private_message = f"{error_type}: {str(e)}"
        return cls(private_message=private_message)
