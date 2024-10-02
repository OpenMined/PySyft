# relative
from ...serde.serializable import serializable
from ...store.db.stash import ObjectStash
from ...util.telemetry import instrument
from .settings import ServerSettings


@instrument
@serializable(canonical_name="SettingsStashSQL", version=1)
class SettingsStash(ObjectStash[ServerSettings]):
    pass
