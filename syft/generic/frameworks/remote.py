class Remote(object):
    frameworks = {}

    def __init__(self, worker, framework_name):
        inst = Remote.frameworks[framework_name](worker)
        setattr(self, inst.name, inst)

    @staticmethod
    def register_framework(cls):
        if cls.name in Remote.frameworks:
            raise AttributeError(f"Error: {cls.name} is in Remote.Frameworks")
        Remote.frameworks[cls.name] = cls
