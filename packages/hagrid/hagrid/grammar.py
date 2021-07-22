# stdlib
import socket
from typing import Any
from typing import Callable
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union


class BadGrammar(Exception):
    pass


class GrammarVerb:
    def __init__(
        self,
        command: str,
        full_sentence: TypeList[TypeDict[str, Any]],
        abbreviations: TypeDict[int, TypeList[str]],
    ) -> None:
        self.command = command
        self.full_sentence = full_sentence
        self.abbreviations = abbreviations


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
        self.type = type
        self.name = name
        self.default = default
        self.options = options
        self.example = example

    def get_example(self) -> str:
        return self.example if self.example else self.default

    # no op
    def custom_parsing(self, input: str) -> str:
        return input

    def parse_input(self, input: Optional[str]) -> str:
        if input is None and self.default is None:
            raise BadGrammar(
                f"{self.name} has no default, please use one of the following options: {self.options}"
            )
        if input is None:
            if isinstance(self.default, str):
                return self.default
            elif isinstance(self.default, Callable):
                return self.default()

        if self.options is not None and input not in self.options:
            raise BadGrammar(
                f"{input} is not valid for {self.name} please use one of the following options: {self.options}"
            )

        return self.custom_parsing(input=input)


class HostGrammarTerm(GrammarTerm):
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
        print("custom s0urce", input)
        # github.com/OpenMined/PySyft/tree/demo_strike_team_branch_4
        parts = input
        if parts.startswith("http://"):
            parts = parts.replace("http://", "")
        if parts.startswith("https://"):
            parts = parts.replace("https://", "")
        if parts.startswith("github.com/"):
            parts = parts.replace("github.com/", "")

        parts = parts.split("/")
        print("parts", parts, len(parts), "tree" not in input)
        if "tree" not in input or len(parts) < 4:
            raise BadGrammar(
                f"{self.name} should be a valid github.com repo branch url. Try: {self.get_example()}"
            )

        repo = f"{parts[0]}/{parts[1]}"
        branch = "/".join(parts[3:])

        print("repo", repo, "branch", branch)
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


def parse_grammar(args: TypeTuple, verb: GrammarVerb) -> str:
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
                term = term_settings["klass"](**term_settings).parse_input(arg)
                terms.append(term)
            except BadGrammar as e:
                errors.append(str(e))

        if len(errors) > 0:
            raise BadGrammar("\n".join(errors))

        # make command
        return " ".join(terms)
    else:
        raise BadGrammar("Grammar is not valid")
