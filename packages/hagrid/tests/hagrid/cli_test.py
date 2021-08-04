# stdlib
from typing import List

# third party
from hagrid import cli


def test_hagrid_launch() -> None:
    """This test is important because we want to make it convenient
    for our developers to just run 'hagrid launch' whenever they want to spin
    up a new node with a randomly chosen name"""

    # COMMAND: "hagrid launch"
    args: List[str] = []

    verb = cli.get_launch_verb()
    grammar = cli.parse_grammar(args=tuple(args), verb=verb)
    verb.load_grammar(grammar=grammar)
    cmd = cli.create_launch_cmd(verb=verb, kwargs={}, ignore_docker_version_check=True)

    # check that it's a domain by default
    assert "NODE_TYPE=domain" in cmd

    # check that the node has a name
    assert "DOMAIN_NAME=" in cmd

    # check that tail is on by default
    assert " -d " not in cmd
