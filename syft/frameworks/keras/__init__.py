import syft

if syft.dependency_check.tfe_available:
    from syft.frameworks.keras import layers  # noqa: F401
    from syft.frameworks.keras import model  # noqa: F401
    from syft.frameworks.keras.hook import KerasHook

    __all__ = [KerasHook.__name__]
