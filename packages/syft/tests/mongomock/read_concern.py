class ReadConcern(object):
    def __init__(self, level=None):
        self._document = {}

        if level is not None:
            self._document["level"] = level

    @property
    def level(self):
        return self._document.get("level")

    @property
    def ok_for_legacy(self):
        return True

    @property
    def document(self):
        return self._document.copy()

    def __eq__(self, other):
        return other.document == self.document
