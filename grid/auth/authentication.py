from abc import ABC


class BaseAuthentication(ABC):
    """ BaseAuthentication abstract class defines generic methods used by all types of authentications defined in this module."""

    def __init__(self, filename):
        self.FILENAME = filename

    def parse(self):
        """ Read, parse and load credential files."""
        raise NotImplementedError("Parse not specified!")

    def json(self):
        """ Convert credential instances into a JSON structure. """
        raise NotImplementedError("JSON not specified!")
