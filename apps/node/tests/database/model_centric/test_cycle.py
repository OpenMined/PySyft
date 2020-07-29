import sys
from random import randint

import pytest
from src.app.main.model_centric.cycles.cycle import Cycle

from . import BIG_INT
from .presets.cycle import CYCLE_ATTRIBUTES

sys.path.append(".")


@pytest.mark.parametrize("start, sequence, end", CYCLE_ATTRIBUTES)
def test_create_cycle_object(start, sequence, end, database):
    new_cycle = Cycle(id=randint(0, BIG_INT), start=start, sequence=sequence, end=end)

    database.session.add(new_cycle)
    database.session.commit()
