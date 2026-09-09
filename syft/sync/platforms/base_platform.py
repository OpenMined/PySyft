from pydantic import BaseModel


class BasePlatform(BaseModel):
    name: str
    module_path: str

    def __hash__(self):
        return hash(self.name)

    @property
    def module_name(self) -> str:
        return self.module_path.split(".")[-1]
