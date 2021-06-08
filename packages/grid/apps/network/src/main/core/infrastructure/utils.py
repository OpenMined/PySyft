# stdlib
from types import SimpleNamespace

# third party
from PyInquirer import Separator
from PyInquirer import Token
from PyInquirer import prompt
from PyInquirer import style_from_dict
import click


class Config(SimpleNamespace):
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return Config(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)


COLORS = SimpleNamespace(
    **{
        "black": "black",
        "red": "red",
        "green": "green",
        "yellow": "yellow",
        "blue": "blue",
        "magenta": "magenta",
        "cyan": "cyan",
        "white": "white",
        "bright_black": "bright_black",
        "bright_red": "bright_red",
        "bright_green": "bright_green",
        "bright_yellow": "bright_yellow",
        "bright_blue": "bright_blue",
        "bright_magenta": "bright_magenta",
        "bright_cyan": "bright_cyan",
        "bright_white": "bright_white",
    }
)


def colored(text, color=COLORS.green, bold=True):
    return click.style(text, fg=color, bold=bold)


styles = SimpleNamespace(
    first=style_from_dict(
        {
            Token.Separator: "#cc5454",
            Token.QuestionMark: "#673ab7",
            Token.Selected: "#cc5454",
            Token.Pointer: "#673ab7 bold",
            Token.Instruction: "",
            Token.Answer: "#f44336 bold",
            Token.Question: "#673ab7",
        }
    ),
    second=style_from_dict(
        {
            Token.Separator: "#6C6C6C",
            Token.QuestionMark: "#FF9D00 bold",
            Token.Selected: "#5F819D",
            Token.Pointer: "#FF9D00 bold",
            Token.Instruction: "",  # default
            Token.Answer: "#5F819D bold",
            Token.Question: "",
        }
    ),
    third=style_from_dict(
        {
            Token.QuestionMark: "#E91E63 bold",
            Token.Selected: "#673AB7 bold",
            Token.Instruction: "",  # default
            Token.Answer: "#2196f3 bold",
            Token.Questions: "",
        }
    ),
)
