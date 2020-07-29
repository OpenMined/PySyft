import pytest
import sys

from . import BIG_INT
from .presets.cycle import CYCLE_ATTRIBUTES

from random import randint
from src.app.main.sfl.cycles.cycle import Cycle

sys.path.append(".")


@pytest.mark.parametrize("start, sequence, end", CYCLE_ATTRIBUTES)
def test_create_cycle_object(start, sequence, end, database):
    new_cycle = Cycle(id=randint(0, BIG_INT), start=start, sequence=sequence, end=end,)

    database.session.add(new_cycle)
    database.session.commit()
