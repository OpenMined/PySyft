from pydantic import BaseModel


class Access(BaseModel):
    """Raw access lists as deserialized from YAML.

    Values can be:
    - exact email: "alice@example.com"
    - domain wildcard: "*@example.com"
    - everyone: "*"
    - placeholder: "USER" (resolved during compilation)
    """

    admin: list[str] = []
    write: list[str] = []
    read: list[str] = []
