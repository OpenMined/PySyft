class _Primary(object):
    @property
    def mongos_mode(self):
        return "primary"

    @property
    def mode(self):
        return 0

    @property
    def name(self):
        return "Primary"

    @property
    def document(self):
        return {"mode": "primary"}

    @property
    def tag_sets(self):
        return [{}]

    @property
    def max_staleness(self):
        return -1

    @property
    def min_wire_version(self):
        return 0


def ensure_read_preference_type(key, value):
    """Raise a TypeError if the value is not a type compatible for ReadPreference."""
    for attr in ("document", "mode", "mongos_mode", "max_staleness"):
        if not hasattr(value, attr):
            raise TypeError(
                "{} must be an instance of {}".format(
                    key, "pymongo.read_preference.ReadPreference"
                )
            )


PRIMARY = _Primary()
