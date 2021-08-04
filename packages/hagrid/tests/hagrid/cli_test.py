# stdlib
from typing import List

# third party
from hagrid import cli


def test_basic_launch() -> None:

    # COMMAND: "hagrid launch"
    args: List[str] = []

    verb = cli.get_launch_verb()
    grammar = cli.parse_grammar(args=tuple(args), verb=verb)
    verb.load_grammar(grammar=grammar)
    cmd = cli.create_launch_cmd(verb=verb, kwargs={})

    # check that it's a domain by default
    assert "NODE_TYPE=domain" in cmd

    # check that the node has a name
    assert "DOMAIN_NAME=" in cmd

    # check that tail is on by default
    assert " -d " not in cmd
