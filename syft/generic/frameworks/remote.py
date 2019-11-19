class Remote(object):
    frameworks = {}

    def __init__(self, worker, framework_name):
        inst = Remote.frameworks[framework_name](worker)
        setattr(self, inst.name, inst)

    @staticmethod
    def register_framework(cls):
        assert cls.name not in Remote.frameworks
        Remote.frameworks[cls.name] = cls
