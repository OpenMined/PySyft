from ....decorators.syft_decorator_impl import syft_decorator


class Location(object):
    """This represents the location of a node, including
    location-relevant metadata (such as how long it takes
    for us to communicate with this location, etc.)"""

    def __init__(self) -> None:
        super().__init__()

    @syft_decorator(typechecking=True)
    def repr_short(self) -> str:
        """Returns a SHORT human-readable version of the ID

        Return a SHORT human-readable version of the ID which
        makes it print nicer when embedded (often alongside other
        UID objects) within other object __repr__ methods."""

        return self.__repr__()
