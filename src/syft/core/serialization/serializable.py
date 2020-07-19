class Serializable:
    @staticmethod
    def serialize(obj: any, protocol: type) -> bin:
        raise NotImplementedError

    @staticmethod
    def deserialize(binary_data: bin, protocol: type) -> any:
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema():
        raise NotImplementedError