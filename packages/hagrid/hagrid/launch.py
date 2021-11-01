# stdlib
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional

# relative
from .cache import DEFAULT_BRANCH
from .grammar import GrammarTerm
from .grammar import GrammarVerb
from .grammar import HostGrammarTerm
from .grammar import SourceGrammarTerm
from .names import random_name


def get_launch_verb() -> GrammarVerb:
    full_sentence = [
        {
            "name": "node_name",
            "type": "adjective",
            "klass": GrammarTerm,
            "default": random_name,
            "example": "'my_domain'",
        },
        {
            "name": "node_type",
            "type": "object",
            "klass": GrammarTerm,
            "default": "domain",
            "options": ["domain", "network"],
        },
        {
            "name": "preposition",
            "type": "preposition",
            "klass": GrammarTerm,
            "default": "to",
            "options": ["to"],
        },
        {
            "name": "host",
            "type": "propernoun",
            "klass": HostGrammarTerm,
            "default": "docker",
            "example": "docker:8081+",
        },
        {
            "name": "preposition",
            "type": "preposition",
            "klass": GrammarTerm,
            "default": "from",
            "options": ["from"],
        },
        {
            "name": "source",
            "type": "propernoun",
            "klass": SourceGrammarTerm,
            "default": f"github.com/OpenMined/PySyft/tree/{DEFAULT_BRANCH}",
        },
    ]

    abbreviations: TypeDict[int, TypeList[Optional[str]]] = {
        6: [
            "adjective",  # name
            "object",  # node_type
            "preposition",  # to
            "propernoun",  # host
            "preposition",  # from
            "propernoun",  # source
        ],
        5: [
            None,  # name
            "object",  # node_type
            "preposition",  # to
            "propernoun",  # host
            "preposition",  # from
            "propernoun",  # source
        ],
        4: [
            "adjective",  # name
            "object",  # node_type
            "preposition",  # to
            "propernoun",  # host
            None,  # ignore
            None,  # ignore
        ],
        3: [
            None,  # ignore
            "object",  # node_type
            "preposition",  # to
            "propernoun",  # host
            None,  # ignore
            None,  # ignore
        ],
        2: [
            "adjective",  # name
            "object",  # node_type
            None,  # ignore
            None,  # ignore
            None,  # ignore
            None,  # ignore
        ],
        1: [
            None,  # ignore
            "object",  # node_type
            None,  # ignore
            None,  # ignore
            None,  # ignore
            None,  # ignore
        ],
    }

    return GrammarVerb(
        command="launch",
        full_sentence=full_sentence,
        abbreviations=abbreviations,
    )
