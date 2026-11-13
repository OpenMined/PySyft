#!/usr/bin/env bash
#
# Gather the review context the syft-pr-feedback skill reads, after
# .claude/skills/syft-pr-review/prefetch.sh has written /tmp/pr-<N>/.
#
# Usage: .claude/skills/syft-pr-feedback/context.sh <pr-number> [--at <commit>]
#
# Writes /tmp/pr-<N>/context/:
#   review-rules.md        docs/engineering/review-rules.md from the base branch
#   invariants.md          the .claude/rules/invariants/ files whose paths match a changed file
#   existing-comments.md   comments already on the PR (none with --at)
#   head.txt               the commit the review is about
#
# Rules and invariants come from the base branch, so a PR cannot change the
# standards of its own review. Until they exist on the base branch, the
# working-tree copies are used, and the index says so.
#
# --at <commit> first runs eval/at_commit.sh, which rewrites the diff to that
# commit, and skips the comments: a back-test must not see the answers.
#
# Read-only with respect to the working tree. Keep this in step with SKILL.md.

set -euo pipefail

PR="${1:?usage: context.sh <pr-number> [--at <commit>]}"
AT=""
if [ "${2:-}" = "--at" ]; then
  AT="${3:?--at needs a commit}"
fi

OUT="/tmp/pr-$PR"
SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git rev-parse --show-toplevel)"

[ -f "$OUT/meta.json" ] || {
  echo "No $OUT/meta.json: run ./.claude/skills/syft-pr-review/prefetch.sh $PR first." >&2
  exit 1
}

if [ -n "$AT" ]; then
  "$SKILL_DIR/eval/at_commit.sh" "$PR" "$AT"
fi

BASE_REF=$(jq -r .baseRefName "$OUT/meta.json")
CTX="$OUT/context"
rm -rf "$CTX"
mkdir -p "$CTX"

if [ -n "$AT" ]; then
  HEAD_SHA=$(git rev-parse "$AT^{commit}")
else
  HEAD_SHA=$(gh pr view "$PR" --json headRefOid -q .headRefOid)
fi
echo "$HEAD_SHA" >"$CTX/head.txt"
BASE_SHA=$(git rev-parse --short "origin/$BASE_REF")

# ------------------------------------------------------------------ rules ---
RULES_PATH=docs/engineering/review-rules.md
if git show "origin/$BASE_REF:$RULES_PATH" >"$CTX/review-rules.md" 2>/dev/null; then
  RULES_FROM="origin/$BASE_REF ($BASE_SHA)"
else
  cp "$REPO/$RULES_PATH" "$CTX/review-rules.md"
  RULES_FROM="the working tree (not on origin/$BASE_REF yet)"
fi

# ------------------------------------------------------------- invariants ---
# Each invariant file starts with YAML front matter that lists `paths:` globs.
# A file is included when one of its globs matches a changed path.
INVARIANTS_FROM=$(python3 - "$OUT" "$BASE_REF" "$REPO" <<'PY'
import fnmatch
import subprocess
import sys
from pathlib import Path

out, base_ref, repo = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
folder = ".claude/rules/invariants"


def git(*args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


listing = git("ls-tree", "-r", "--name-only", f"origin/{base_ref}", "--", folder)
names = [n for n in listing.stdout.split() if n.endswith(".md")]
if names:
    source = f"origin/{base_ref}"
    files = {n: git("show", f"origin/{base_ref}:{n}").stdout for n in names}
else:
    source = "the working tree (not on the base branch yet)"
    files = {
        str(p.relative_to(repo)): p.read_text()
        for p in sorted((repo / folder).glob("*.md"))
    }

# diff-index.txt comes from the full diff. files.txt does not do here: it comes
# from `gh pr view --json files`, which stops at 100 files.
changed = [
    line.split("\t", 1)[0]
    for line in (out / "diff-index.txt").read_text().splitlines()
    if line.strip()
]


def globs(text):
    """The `paths:` list of the front matter, as written."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    found, inside = [], False
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("paths:"):
            inside = True
            continue
        if inside and line.lstrip().startswith("- "):
            found.append(line.split("- ", 1)[1].strip().strip("\"'"))
        elif inside and line and not line.startswith(" "):
            inside = False
    return found


def matches(path, pattern):
    # fnmatch's * also matches "/", so "a/**" matches everything under a/.
    # "**/x" must also match a top-level "x".
    return fnmatch.fnmatch(path, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:])
    )


picked = []
for name, text in files.items():
    hits = sorted({p for p in changed for g in globs(text) if matches(p, g)})
    if hits:
        picked.append((name, text, hits))

with open(out / "context" / "invariants.md", "w") as f:
    if not picked:
        f.write("No invariant file covers the changed paths.\n")
    for name, text, hits in picked:
        f.write(f"<!-- {name}: matched {len(hits)} changed file(s), e.g. {hits[0]} -->\n")
        body = text.split("---", 2)[2] if text.startswith("---") else text
        f.write(body.strip() + "\n\n")

print(f"{source}: {len(picked)} of {len(files)} file(s) match")
PY
)

# --------------------------------------------------------------- comments ---
if [ -n "$AT" ]; then
  echo "Back-test at $AT: comments are not read." >"$CTX/existing-comments.md"
else
  {
    echo "# Comments already on PR $PR"
    echo
    echo "## Review summaries"
    gh api --paginate "repos/{owner}/{repo}/pulls/$PR/reviews" \
      --jq '.[] | select(.body != "") | "- \(.user.login) (\(.state)): \(.body | gsub("\\s+"; " ") | .[0:400])"'
    echo
    echo "## Inline comments"
    gh api --paginate "repos/{owner}/{repo}/pulls/$PR/comments" \
      --jq '.[] | "- \(.user.login) on \(.path):\(.line // .original_line): \(.body | gsub("\\s+"; " ") | .[0:400])"'
    echo
    echo "## Conversation"
    gh api --paginate "repos/{owner}/{repo}/issues/$PR/comments" \
      --jq '.[] | "- \(.user.login): \(.body | gsub("\\s+"; " ") | .[0:400])"'
  } >"$CTX/existing-comments.md"
fi

# ------------------------------------------------------------------ index ---
cat <<SUMMARY
context for PR $PR -> $CTX

  head.txt               $(cut -c1-10 "$CTX/head.txt")${AT:+  (back-test: the PR as it was at this commit)}
  review-rules.md        from $RULES_FROM
  invariants.md          from $INVARIANTS_FROM
  existing-comments.md   $(grep -c '^- ' "$CTX/existing-comments.md" || true) comment(s)
SUMMARY
