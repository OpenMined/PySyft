# future
from __future__ import annotations

# stdlib
import socket
from typing import Any
from typing import Callable
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union

# relative
from .lib import find_available_port


class BadGrammar(Exception):
    pass


class GrammarVerb:
    def __init__(
        self,
        command: str,
        full_sentence: TypeList[TypeDict[str, Any]],
        abbreviations: TypeDict[int, TypeList[str]],
    ) -> None:
        self.grammar = None
        self.command = command
        self.full_sentence = full_sentence
        self.abbreviations = abbreviations

    def get_named_term_type(
        self, name: str, term_type: Optional[str] = None
    ) -> Optional[GrammarTerm]:
        found = None
        for term in self.grammar:
            if term.name == name:
                if term_type is not None and term.type == term_type:
                    found = term
                elif term_type is None:
                    found = term
        return found

    def set_named_term_type(
        self, name: str, new_term: GrammarTerm, term_type: Optional[str] = None
    ) -> None:
        new_grammar = []
        for term in self.grammar:
            found = False
            if term.name == name:
                if term_type is not None and term.type == term_type:
                    found = True
                elif term_type is None:
                    found = True
            if not found:
                new_grammar.append(term)
            else:
                new_grammar.append(new_term)
        self.grammar = new_grammar

    def load_grammar(self, grammar: TypeList[GrammarTerm]) -> None:
        self.grammar = grammar


class GrammarTerm:
    def __init__(
        self,
        type: str,
        name: str,
        default: Optional[Union[str, Callable]] = None,
        options: Optional[TypeList] = None,
        example: Optional[str] = None,
        **kwargs: TypeDict[str, Any],
    ) -> None:
        self.raw_input = None
        self.input = None
        self.type = type
        self.name = name
        self.default = default
        self.options = options
        self.example = example

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name}<{self.type}>: {self.input} [raw: {self.raw_input}]>"

    def get_example(self) -> str:
        return self.example if self.example else self.default

    # no op
    def custom_parsing(self, input: str) -> str:
        return input

    def parse_input(self, input: Optional[str]) -> None:
        self.raw_input = input
        if input is None and self.default is None:
            raise BadGrammar(
                f"{self.name} has no default, please use one of the following options: {self.options}"
            )
        if input is None:
            if isinstance(self.default, str):
                input = self.default
            elif isinstance(self.default, Callable):
                input = self.default()

        if self.options is not None and input not in self.options:
            raise BadGrammar(
                f"{input} is not valid for {self.name} please use one of the following options: {self.options}"
            )

        self.input = self.custom_parsing(input=input)


class HostGrammarTerm(GrammarTerm):
    @property
    def host(self) -> Optional[str]:
        return self.parts()[0]

    @property
    def port(self) -> Optional[str]:
        return self.parts()[1]

    @property
    def search(self) -> bool:
        return self.parts()[2]

    @property
    def free_port(self) -> int:
        return find_available_port(host="localhost", port=self.port, search=self.search)

    def parts(self) -> TypeTuple[Optional[str], Optional[int], Optional[bool]]:
        host = None
        port = None
        search = False
        if self.input:
            parts = self.input.split(":")
            host = parts[0]
            port = parts[1]
            if port.endswith("+"):
                search = True
                port = port[0:-1]
            port = int(port)
        return (host, port, search)

    def validate_host(self, host_or_ip: str) -> bool:
        try:
            if socket.gethostbyname(host_or_ip) == host_or_ip:
                return True
            elif socket.gethostbyname(host_or_ip) != host_or_ip:
                return True
        except socket.gaierror:
            raise BadGrammar(
                f"{host_or_ip} is not valid for {self.name}. Try an IP, hostname or docker, vm, aws, azure or gcp"
            )
        return False

    def validate_port(self, port: str) -> bool:
        try:
            if port.endswith("+"):
                int(port[0:-1])
            else:
                int(port)
        except Exception:
            raise BadGrammar(
                f"{port} is not a valid port option. Try: {self.get_example()}"
            )
        return True

    def custom_parsing(self, input: Optional[str]) -> str:
        colons = input.count(":")
        host = input
        port = None
        if colons > 1:
            raise BadGrammar(
                f"You cannot have more than one : for {self.name}, try: {self.get_example()}"
            )
        elif colons == 1:
            parts = input.split(":")
            host = parts[0]
            port = parts[1]

        if port is None:
            if host == "docker":
                port = "8081+"  # default
            else:
                port = "80"  # default

        if host not in ["docker", "vm", "aws", "azure", "gcp"]:
            _ = self.validate_host(host_or_ip=host)

        _ = self.validate_port(port=port)

        return f"{host}:{port}"


class SourceGrammarTerm(GrammarTerm):
    def custom_parsing(self, input: Optional[str]) -> str:
        # github.com/OpenMined/PySyft/tree/demo_strike_team_branch_4
        parts = input
        if parts.startswith("http://"):
            parts = parts.replace("http://", "")
        if parts.startswith("https://"):
            parts = parts.replace("https://", "")
        if parts.startswith("github.com/"):
            parts = parts.replace("github.com/", "")

        parts = parts.split("/")
        if "tree" not in input or len(parts) < 4:
            raise BadGrammar(
                f"{self.name} should be a valid github.com repo branch url. Try: {self.get_example()}"
            )

        repo = f"{parts[0]}/{parts[1]}"
        branch = "/".join(parts[3:])

        return f"{repo}:{branch}"


def validate_arg_count(arg_count: int, verb: GrammarVerb) -> bool:
    valid = True

    if arg_count not in verb.abbreviations:
        error_str = f"Command {verb.command} supports the following invocations:\n"
        for count in sorted(verb.abbreviations.keys()):
            abbreviation = verb.abbreviations[count]
            example_terms = []
            for i, term_type in enumerate(abbreviation):
                if term_type is not None:
                    term_settings = verb.full_sentence[i]
                    example = term_settings["klass"](**term_settings).get_example()
                    example_terms.append(example)
            error_str += f"{count} args: {verb.command} {' '.join(example_terms)}\n"

        raise BadGrammar(error_str)

    return valid


def parse_grammar(args: TypeTuple, verb: GrammarVerb) -> TypeList[GrammarTerm]:
    arg_list = list(args)
    arg_count = len(arg_list)
    errors = []
    if validate_arg_count(arg_count=arg_count, verb=verb):
        terms = []
        abbreviation = verb.abbreviations[arg_count]
        for i, term_type in enumerate(abbreviation):
            if term_type is None:
                arg = None  # use None so we get the default
            else:
                arg = arg_list.pop(0)  # use a real arg
            term_settings = verb.full_sentence[i]

            try:
                term = term_settings["klass"](**term_settings)
                term.parse_input(arg)
                terms.append(term)
            except BadGrammar as e:
                errors.append(str(e))

        if len(errors) > 0:
            raise BadGrammar("\n".join(errors))

        # make command
        return terms
    else:
        raise BadGrammar("Grammar is not valid")
