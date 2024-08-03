class _Primary:
    @property
    def mongos_mode(self) -> str:
        return "primary"

    @property
    def mode(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return "Primary"

    @property
    def document(self):
        return {"mode": "primary"}

    @property
    def tag_sets(self):
        return [{}]

    @property
    def max_staleness(self) -> int:
        return -1

    @property
    def min_wire_version(self) -> int:
        return 0


def ensure_read_preference_type(key, value) -> None:
    """Raise a TypeError if the value is not a type compatible for ReadPreference."""
    for attr in ("document", "mode", "mongos_mode", "max_staleness"):
        if not hasattr(value, attr):
            msg = (
                "{} must be an instance of {}".format(
                    key, "pymongo.read_preference.ReadPreference",
                )
            )
            raise TypeError(
                msg,
            )


PRIMARY = _Primary()
