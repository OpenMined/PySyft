# stdlib

# relative
from .grammar import GrammarTerm
from .grammar import GrammarVerb
from .grammar import HostGrammarTerm


def get_land_verb() -> GrammarVerb:
    full_sentence = [
        {
            "name": "node_name",
            "type": "adjective",
            "klass": GrammarTerm,
            "example": "'my_domain'",
        },
        {
            "name": "preposition",
            "type": "preposition",
            "klass": GrammarTerm,
            "default": "at",
            "options": ["at", "on"],
        },
        {
            "name": "host",
            "type": "propernoun",
            "klass": HostGrammarTerm,
            "default": "docker",
            "example": "docker",
        },
    ]

    abbreviations: dict[int, list[str | None]] = {
        3: [
            "adjective",
            "preposition",
            "propernoun",
        ],  # node_name  # at  # host
        2: [
            "adjective",
            None,
            "propernoun",
        ],  # node_name  # ignore  # host
        1: [
            "adjective",
            None,
            None,
        ],  # node_name  # ignore  # ignore
        0: [
            None,
            None,
            None,
        ],  # ignore  # ignore  # ignore
    }

    return GrammarVerb(
        command="land",
        full_sentence=full_sentence,
        abbreviations=abbreviations,
    )
