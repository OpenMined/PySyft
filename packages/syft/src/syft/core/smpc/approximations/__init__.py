"""Approximation Functions."""
# relative
from .exp import exp
from .log import log
from .reciprocal import reciprocal
from .utils import sign

APPROXIMATIONS = {
    "log": log,
    "exp": exp,
    "reciprocal": reciprocal,
    "sign": sign,
}

__all__ = ["APPROXIMATIONS"]
