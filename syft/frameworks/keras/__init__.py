import syft

if syft.dependency_check.tfe_available:
    from syft.frameworks.keras import layers
    from syft.frameworks.keras import model
    from syft.frameworks.keras.hook import KerasHook

    __all__ = ["KerasHook"]
