import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class PydanticFormatter(ABC):
    """Interface for Pydantic model formatters, to be used with PydanticFormatterMixin"""

    @abstractmethod
    def format_str(self, model: BaseModel) -> str: ...

    @abstractmethod
    def format_repr(self, model: BaseModel) -> str: ...

    def format_markdown(self, model: BaseModel) -> str | None:
        return None

    def format_html(self, model: BaseModel) -> str | None:
        return None


class ANSIPydanticFormatter(PydanticFormatter):
    """Format Pydantic models multiline string with ANSI colors"""

    def __init__(self):
        self.use_colors = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def format_class_name(self, name: str) -> str:
        if self.use_colors:
            return f"\033[1;36m{name}\033[0m"
        return name

    def format_field(self, key: str, value: Any) -> str:
        value_str = str(value)
        formatted_key = key
        formatted_value = value_str
        if self.use_colors:
            formatted_key = f"\033[1m{key}\033[0m"

            if isinstance(value, (int, float)):
                formatted_value = f"\033[36m{value_str}\033[0m"
            elif isinstance(value, str):
                formatted_value = f"\033[32m{value_str}\033[0m"
            elif isinstance(value, (list, dict)):
                formatted_value = f"\033[33m{value_str}\033[0m"
            else:
                formatted_value = f"\033[34m{value_str}\033[0m"
        return f"  {formatted_key}: {formatted_value}"

    def format_str(self, model: BaseModel) -> str:
        header = self.format_class_name(model.__class__.__name__) + "\n"

        fields = model.model_dump(mode="json")
        items = [self.format_field(k, v) for k, v in fields.items()]

        return header + "\n".join(items)

    def format_repr(self, model: BaseModel) -> str:
        return self.format_str(model)


class PydanticFormatterMixin:
    __display_formatter__: ClassVar[PydanticFormatter] = ANSIPydanticFormatter()

    def __str__(self) -> str:
        return self.__display_formatter__.format_str(self)

    def __repr__(self) -> str:
        return self.__display_formatter__.format_repr(self)

    def _repr_html_(self) -> str:
        return self.__display_formatter__.format_html(self)

    def _repr_markdown_(self) -> str:
        return self.__display_formatter__.format_markdown(self)
