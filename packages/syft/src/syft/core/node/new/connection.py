class NodeConnection:
    def get_cache_key() -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}"
