"""Peer-related exceptions for syft."""


class SyftPeerNotReadyError(ValueError):
    """A peer is known but not yet usable for the operation.

    Carries the cause and the remedy separately so callers — a readiness
    helper, or a diagnostic — can show either one on its own.

    Subclasses ValueError because these conditions were reported as a bare
    ValueError before, and callers catch that.
    """

    def __init__(self, peer_email: str, cause: str, remedy: str):
        self.peer_email = peer_email
        self.cause = cause
        self.remedy = remedy
        super().__init__(f"{cause} {remedy}")
