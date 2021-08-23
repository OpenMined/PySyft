# stdlib
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional

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
            "options": ["at"],
        },
        {
            "name": "host",
            "type": "propernoun",
            "klass": HostGrammarTerm,
            "default": "docker",
            "example": "docker",
        },
    ]

    abbreviations: TypeDict[int, TypeList[Optional[str]]] = {
        3: [
            "adjective",  # node_name
            "preposition",  # at
            "propernoun",  # host
        ],
        2: [
            "adjective",  # node_name
            None,  # ignore
            "propernoun",  # host
        ],
        1: [
            "adjective",  # node_name
            None,  # ignore
            None,  # ignore
        ],
        0: [
            None,  # ignore
            None,  # ignore
            None,  # ignore
        ],
    }

    return GrammarVerb(
        command="land",
        full_sentence=full_sentence,
        abbreviations=abbreviations,
    )
