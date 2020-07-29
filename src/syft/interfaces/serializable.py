class Serializable:
    @staticmethod
    def to_protobuf(self):
        raise NotImplementedError

    @staticmethod
    def from_protobuf(proto):
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema():
        raise NotImplementedError

    @staticmethod
    def get_wrapped_type():
        raise NotImplementedError
