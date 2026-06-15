from dataclasses import dataclass, field

from syft_permissions.engine.compiled_rule import ACLRule, compile_rule
from syft_permissions.engine.request import ACLRequest
from syft_permissions.engine.utils import specificity_key
from syft_permissions.spec.ruleset import RuleSet


@dataclass
class ACLNode:
    children: dict[str, "ACLNode"] = field(default_factory=dict)
    ruleset: RuleSet | None = None
    terminal: bool = False
    path: str = ""


class ACLTree:
    def __init__(self):
        self.root = ACLNode(path="")

    def add_ruleset(self, rs: RuleSet) -> None:
        segments = _path_segments(rs.path)
        node = self.root
        for seg in segments:
            if seg not in node.children:
                node.children[seg] = ACLNode()
            node = node.children[seg]
        node.ruleset = rs
        node.terminal = rs.terminal
        node.path = rs.path
        # Pre-sort rules by specificity (most specific first)
        if rs.rules:
            rs.rules.sort(key=lambda r: specificity_key(r.pattern), reverse=True)

    def remove_ruleset(self, path: str) -> None:
        segments = _path_segments(path)
        node = self.root
        for seg in segments:
            if seg not in node.children:
                return
            node = node.children[seg]
        node.ruleset = None
        node.terminal = False

    def get_nearest_node(self, path: str) -> ACLNode | None:
        """Walk trie toward target. Return deepest node that has rules.
        Stop early at terminal nodes."""
        segments = _path_segments(path)
        node = self.root
        nearest = node if node.ruleset else None

        for seg in segments:
            if node.terminal and node.ruleset:
                return node
            if seg not in node.children:
                break
            node = node.children[seg]
            if node.ruleset:
                nearest = node

        return nearest

    def get_compiled_rule(self, request: ACLRequest) -> ACLRule | None:
        """Find the nearest node and return the first matching compiled rule."""
        node = self.get_nearest_node(request.path)
        if node is None or node.ruleset is None:
            return None

        # Compute relative path from node's directory to the target
        rel_path = _relative_path(node.path, request.path)
        user = request.user.id

        # rules are already sorted, so we can return the first match
        for rule in node.ruleset.rules:
            compiled = compile_rule(rule, user)
            if compiled.match(rel_path, user):
                return compiled

        return None


def _path_segments(path: str) -> list[str]:
    """Split a path into non-empty segments."""
    return [s for s in path.strip("/").split("/") if s]


def _relative_path(base: str, target: str) -> str:
    """Compute path of target relative to base directory."""
    base_parts = _path_segments(base)
    target_parts = _path_segments(target)

    if not base_parts:
        return "/".join(target_parts)

    # Strip the base prefix from target
    if target_parts[: len(base_parts)] == base_parts:
        remaining = target_parts[len(base_parts) :]
        return "/".join(remaining) if remaining else ""

    return "/".join(target_parts)
