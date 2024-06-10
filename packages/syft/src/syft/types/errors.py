class SyftException(Exception):
    def public(self) -> str:
        return "An error occurred. Contact your admininstrator for more information."

