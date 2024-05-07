def _with_default_values(document):
    if "w" in document:
        return document
    return dict(document, w=1)


class WriteConcern(object):
    def __init__(self, w=None, wtimeout=None, j=None, fsync=None):
        self._document = {}
        if w is not None:
            self._document["w"] = w
        if wtimeout is not None:
            self._document["wtimeout"] = wtimeout
        if j is not None:
            self._document["j"] = j
        if fsync is not None:
            self._document["fsync"] = fsync

    def __eq__(self, other):
        try:
            return _with_default_values(other.document) == _with_default_values(
                self.document
            )
        except AttributeError:
            return NotImplemented

    def __ne__(self, other):
        try:
            return _with_default_values(other.document) != _with_default_values(
                self.document
            )
        except AttributeError:
            return NotImplemented

    @property
    def acknowledged(self):
        return True

    @property
    def document(self):
        return self._document.copy()

    @property
    def is_server_default(self):
        return not self._document
