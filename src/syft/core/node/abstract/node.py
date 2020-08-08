from ...io.location import Location


class AbstractNode(Location):
    """This only exists to prevent circular dependencies.

    DO NOT ADD FUNCTIONALITY TO THIS CLASS!!! THIS CLASS IS ONLY SUBCLASSED BY ONE OTHER

    If you are trying to subclass this directly - you're doing something wrong."""


class AbstractNodeClient:
    """This only exists to prevent circular dependencies.

    DO NOT ADD FUNCTIONALITY TO THIS CLASS!!! THIS CLASS IS ONLY SUBCLASSED BY ONE OTHER

    If you're trying to subclass this directly - you're doing something wrong."""
