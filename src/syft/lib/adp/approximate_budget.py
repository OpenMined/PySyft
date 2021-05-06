class ApproximateBudget:
    value: int = 0

    def __repr__(self) -> str:
        return f"{type(self)}: {self.value}"
