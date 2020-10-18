from abc import ABC


class AbstractPlanTranslator(ABC):
    """
    Translator class takes a Plan and makes copy that is translated
    to different Plan type, e.g. torchscript
    """

    def __init__(self, plan):
        self.plan = plan

    def translate(self):
        pass

    def remove(self):
        pass
