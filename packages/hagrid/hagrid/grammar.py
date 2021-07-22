# stdlib
from typing import Any
from typing import Callable
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union

"""
launch_grammar = [
        {"type": "verb", "command": "launch"]},
        {"type": "adjective", "default": random_name},
        {"type": "object", "default": "domain", "options": ["domain", "network"]},
        {"type": "preposition", "default": "to", "options": ["to"]},
        {"type": "propernoun", "default": "docker"},
        {"type": "preposition", "default": "from", "options": ["from"]},
        {
            "type": "propernoun",
            "default": "github.com/OpenMined/PySyft/tree/demo_strike_team_branch_4",
        },
    ]
"""


class BadGrammar(Exception):
    pass


class GrammarVerb:
    def __init__(
        self, type: str, command: str, mappings: TypeDict[int, TypeList[str]]
    ) -> None:
        self.type = type
        self.command = command
        self.mappings = mappings


class GrammarTerm:
    def __init__(
        self,
        type: str,
        default: Optional[Union[str, Callable]],
        options: Optional[TypeList],
        example: Optional[str],
    ) -> None:
        self.type = type
        self.default = default
        self.options = options
        self.example = example

    def get_example(self) -> str:
        return self.example if self.example else self.default

    def parse_input(self, input: Optional[str]) -> str:
        if input is None and self.default is None:
            raise BadGrammar(
                f"{self.type} has no default, please use one of the following options: {self.options}"
            )
        if input is None:
            if isinstance(self.default, str):
                return self.default
            elif isinstance(self.default, Callable):
                return self.default()

        return input


def parse_grammar(args: TypeTuple, grammar: TypeList[TypeDict[str, Any]]) -> TypeList:
    print("parsing grammar")
    print(len(args), args)

    terms = []
    verb = GrammarVerb(**grammar.pop(0))
    if len(args) not in verb.mappings:
        error_str = f"Command {verb.command} supports the following invocations:\n"
        for count in sorted(verb.mappings.keys()):
            order = verb[count]
            error_str += f"{count} args: {verb.command} {' '.join(order)}\n"

        raise BadGrammar(error_str)
    print("working with", verb)
    for i, arg in enumerate(args):
        terms.append(GrammarTerm(**grammar[i]).parse_input(arg))

    print("terms", terms)

    # if len(args) > len(grammar):
    #     raise BadGrammar(f"Command {grammar[0][""]}You have used {len(args)} arguments on ")
