# stdlib
from collections import defaultdict
from typing import List
from typing import Tuple

# third party
from hagrid import cli
from hagrid import grammar


def test_hagrid_launch() -> None:
    """This test is important because we want to make it convenient
    for our users to just run 'hagrid launch' whenever they want to spin
    up a new node with a randomly chosen name"""

    # COMMAND: "hagrid launch"
    args: List[str] = []

    verb = cli.get_launch_verb()
    grammar = cli.parse_grammar(args=tuple(args), verb=verb)
    verb.load_grammar(grammar=grammar)
    cmd = cli.create_launch_cmd(
        verb=verb, kwargs=defaultdict(lambda: None), ignore_docker_version_check=True
    )

    cmd = cmd["Launching"][0]  # type: ignore

    # check that it's a domain by default
    assert "NODE_TYPE=domain" in cmd or "NODE_TYPE='domain'" in cmd

    # check that the node has a name
    assert "DOMAIN_NAME=" in cmd

    # check that tail is on by default
    assert " -d " not in cmd


def test_shortand_parse() -> None:
    """This test is important because we want to make it convenient
    for our users to just run 'hagrid launch' whenever they want to spin
    up a new node with a randomly chosen name."""

    # COMMAND: "hagrid launch"
    args: Tuple = ()
    args = grammar.launch_shorthand_support(args)

    # check that domain gets added to the end of the command
    assert args == ("domain",)


def test_hagrid_launch_without_name_with_preposition() -> None:
    """This test is important because we want to make it convenient
    for our users to just run 'hagrid launch' whenever they want to spin
    up a new node with a randomly chosen name"""

    # COMMAND: "hagrid launch on docker"
    args: List[str] = ["to", "docker"]

    verb = cli.get_launch_verb()
    grammar = cli.parse_grammar(args=tuple(args), verb=verb)
    verb.load_grammar(grammar=grammar)
    cmd = cli.create_launch_cmd(
        verb=verb, kwargs=defaultdict(lambda: None), ignore_docker_version_check=True
    )

    cmd = cmd["Launching"][0]  # type: ignore
    # check that it's a domain by default
    assert "NODE_TYPE=domain" in cmd or "NODE_TYPE='domain'" in cmd

    # check that the node has a name
    assert "DOMAIN_NAME=" in cmd

    # check that tail is on by default
    assert " -d " not in cmd


def test_shortand_parse_without_name_with_preposition() -> None:
    """This test is important because we want to make it convenient
    for our users to just run 'hagrid launch' whenever they want to spin
    up a new node with a randomly chosen name."""

    # COMMAND: "hagrid launch"
    args: Tuple[str, ...] = tuple(["to", "docker"])
    args = grammar.launch_shorthand_support(args)

    # check that domain gets added to the end of the command
    assert args == ("domain", "to", "docker")


def test_launch_with_multiword_domain_name() -> None:
    """This test is important because we want to make it convenient
    for our users to just run 'hagrid launch' whenever they want to spin
    up a new node with a randomly chosen name"""

    # COMMAND: "hagrid launch United Nations"
    args: List[str] = ["United", "Nations"]

    verb = cli.get_launch_verb()
    grammar = cli.parse_grammar(args=tuple(args), verb=verb)
    verb.load_grammar(grammar=grammar)
    cmd = cli.create_launch_cmd(
        verb=verb, kwargs=defaultdict(lambda: None), ignore_docker_version_check=True
    )

    cmd = cmd["Launching"][0]  # type: ignore

    # check that it's a domain by default
    assert "NODE_TYPE=domain" in cmd or "NODE_TYPE='domain'" in cmd

    # check that the node has a name
    assert "DOMAIN_NAME=united_nations" in cmd or "DOMAIN_NAME='united_nations'" in cmd

    # check that tail is on by default
    assert " -d " not in cmd


def test_launch_with_longer_multiword_domain_name() -> None:
    """This test is important because we want to make it convenient for users to launch nodes with
    an arbitrary number of words."""

    # COMMAND: "hagrid launch United Nations"
    args: List[str] = ["United", "States", "of", "America"]

    verb = cli.get_launch_verb()
    grammar = cli.parse_grammar(args=tuple(args), verb=verb)
    verb.load_grammar(grammar=grammar)
    cmd = cli.create_launch_cmd(
        verb=verb, kwargs=defaultdict(lambda: None), ignore_docker_version_check=True
    )

    cmd = cmd["Launching"][0]  # type: ignore

    # check that it's a domain by default
    assert "NODE_TYPE=domain" in cmd or "NODE_TYPE='domain'" in cmd

    # check that the node has a name
    assert (
        "DOMAIN_NAME=united_states_of_america" in cmd
        or "DOMAIN_NAME='united_states_of_america'" in cmd
    )

    # check that tail is on by default
    assert " -d " not in cmd


def test_launch_with_longer_multiword_domain_name_with_preposition() -> None:
    """This test is important because we want to make it convenient for users to launch nodes with
    an arbitrary number of words."""

    # COMMAND: "hagrid launch United Nations on docker"
    args: List[str] = ["United", "Nations", "to", "docker"]

    verb = cli.get_launch_verb()
    grammar = cli.parse_grammar(args=tuple(args), verb=verb)
    verb.load_grammar(grammar=grammar)
    cmd = cli.create_launch_cmd(
        verb=verb, kwargs=defaultdict(lambda: None), ignore_docker_version_check=True
    )

    cmd = cmd["Launching"][0]  # type: ignore

    # check that it's a domain by default
    assert "NODE_TYPE=domain" in cmd or "NODE_TYPE='domain'" in cmd

    # check that the node has a name
    assert "DOMAIN_NAME=united_nations" in cmd or "DOMAIN_NAME='united_nations'" in cmd

    # check that tail is on by default
    assert " -d " not in cmd


def test_shortand_parse_of_multiword_name() -> None:
    """This test is important because we want to make it convenient
    for our users to just run 'hagrid launch Multiple Word Name Of Node' whenever they want to spin
    up a new node with a name that has multiple words."""

    # COMMAND: "hagrid launch"
    args: Tuple[str, ...] = tuple(["United", "Nations"])
    args = grammar.launch_shorthand_support(args)

    # check that domain gets added to the end of the command
    assert args == (
        "United Nations",
        "domain",
    )


def test_shortand_parse_of_multiword_name_with_domain() -> None:
    """This test is important because we want to make it convenient
    for our users to just run 'hagrid launch Multiple Word Name Of Node' whenever they want to spin
    up a new node with a name that has multiple words."""

    # COMMAND: "hagrid launch"
    args: Tuple[str, ...] = tuple(["United", "Nations", "domain"])
    args = grammar.launch_shorthand_support(args)

    # check that domain gets added to the end of the command
    assert args == (
        "United Nations",
        "domain",
    )
