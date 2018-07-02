import importlib
import torch


class BaseHook(object):
    r""" A abstract interface for deep learning framework hooks."""

    def __init__(self):
        ""

    def __enter__(self):
        pass

    def __exit__(self):
        importlib.reload(torch)
