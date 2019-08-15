import syft

if syft.dependency_check.tfe_available:
    from . import model
    from . import layers
    from syft.frameworks.keras.hook import KerasHook

    __all__ = ["KerasHook"]
