import numpy as np
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT


def generate_data(rows: int, columns: int, lower_bound: int, upper_bound: int) -> np.array:
    return np.random.randint(lower_bound, upper_bound, size=(rows, columns), dtype=np.int32)


def generate_entity() -> Entity:
    return Entity(name="Ishan")


def make_bounds(data, bound: int) -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly
    generated b/w 0-1"""
    return np.ones_like(data) * bound


def make_sept(np_data: np.array, upper_bound, lower_bound) -> SEPT:
    return SEPT(
        child=np_data,
        entity=generate_entity(),
        max_vals=make_bounds(np_data, upper_bound),
        min_vals=make_bounds(np_data, lower_bound),
    )
