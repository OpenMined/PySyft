import pytest


def test_get_hooks():
    from syft.frameworks.keras import get_hooks
    out = get_hooks(torch=None, keras=None)
