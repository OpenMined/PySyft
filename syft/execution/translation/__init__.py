from enum import Enum


class TranslationTarget(Enum):
    PYTORCH = "torch"
    TENSORFLOW = "tf"
    TENSORFLOW_JS = "tfjs"
    TORCHSCRIPT = "torchscript"

    @staticmethod
    def list():
        return list(map(lambda v: v.value, TranslationTarget))
