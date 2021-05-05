# stdlib
import os
import sys
from typing import Dict

# third party
import config.settings

# create settings object corresponding to specified env
# Search by preset Op. System environment $APP_ENV, if it's not defined
# return Dev as a default value (It'll make the app run in Development mode).
APP_ENV = os.environ.get("APP_ENV", "Dev")

# Search by the constant settings class that represents the current environment.
# Examples:
# APP_ENV == Dev -> config.settings.DevConfig will be chosen.
# APP_ENV == Production -> config.settings.ProductionConfig will be chosen
# APP_ENV == Test -> config.settings.TestConfig will be chosen
_current = getattr(sys.modules["config.settings"], "{0}Config".format(APP_ENV))()

# Copy Chosen Settings class attributes and values
# in order to setup the desired flask environment.
for atr in [f for f in dir(_current) if not "__" in f]:
    # environment can override anything
    val = os.environ.get(atr, getattr(_current, atr))
    setattr(sys.modules[__name__], atr, val)


def as_dict() -> Dict:
    res = {}
    for atr in [f for f in dir(config) if not "__" in f]:
        val = getattr(config, atr)
        res[atr] = val
    return res
