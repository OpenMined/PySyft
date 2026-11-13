#!/usr/bin/env bash
#
# For back-tests: make /tmp/pr-<N>/ describe the PR as it was at <commit>, so
# a review sees only what a human reviewer saw at that commit.
#
# Usage: .claude/skills/syft-pr-feedback/eval/at_commit.sh <pr-number> <commit>
#
# Run after .claude/skills/syft-pr-review/prefetch.sh <N>; context.sh --at runs
# this for you. It rewrites what prefetch.sh derived from the PR's final head:
# full.diff, diff/, diff-index.txt, files.txt, files-skippable.txt, commits.txt,
# merges.txt, new-symbols.txt, new-tests.txt, and the size fields of meta.json.
# Later commits are left out, because their messages describe the fixes.
#
# The splitting and symbol steps copy prefetch.sh. Keep the two in step.

set -euo pipefail

PR="${1:?usage: at_commit.sh <pr-number> <commit>}"
AT="${2:?usage: at_commit.sh <pr-number> <commit>}"
OUT="/tmp/pr-$PR"

[ -f "$OUT/meta.json" ] || { echo "Run prefetch.sh $PR first." >&2; exit 1; }

have() { git cat-file -e "$1^{commit}" 2>/dev/null; }
have "$AT" || git fetch -q origin "$AT"
AT=$(git rev-parse "$AT^{commit}")

# The base the reviewer saw: where the PR branch left its base branch. For a
# merged PR, that is the merge base with the base branch just before the merge
# (the merge commit's first parent), because the base branch now contains the
# PR itself.
BASE_REF=$(jq -r .baseRefName "$OUT/meta.json")
MERGE=$(gh pr view "$PR" --json mergeCommit -q '.mergeCommit.oid // empty')
if [ -n "$MERGE" ]; then
  have "$MERGE" || git fetch -q origin "$MERGE"
  BASE=$(git merge-base "$AT" "$MERGE^1")
else
  BASE=$(git merge-base "$AT" "origin/$BASE_REF")
fi

git diff "$BASE" "$AT" >"$OUT/full.diff"

git diff --numstat "$BASE" "$AT" \
  | awk -F'\t' '{print $1 "+ " $2 "- " $3}' | sort -rn >"$OUT/files.txt"
grep -E '(\.lock|lock\.json|\.ipynb|\.min\.(js|css)|\.svg|\.png|\.csv|\.parquet)$|(^| )\.claude/' \
  "$OUT/files.txt" >"$OUT/files-skippable.txt" || true

git log --oneline --no-merges "$BASE..$AT" >"$OUT/commits.txt"
git log --oneline --merges "$BASE..$AT" >"$OUT/merges.txt"

ADD=$(git diff --numstat "$BASE" "$AT" | awk '$1 != "-" {s += $1} END {print s + 0}')
DEL=$(git diff --numstat "$BASE" "$AT" | awk '$2 != "-" {s += $2} END {print s + 0}')
NFILES=$(wc -l <"$OUT/files.txt" | tr -d ' ')
jq --argjson a "$ADD" --argjson d "$DEL" --argjson n "$NFILES" --arg at "$AT" \
  '.additions = $a | .deletions = $d | .changedFiles = $n | .reviewedAt = $at' \
  "$OUT/meta.json" >"$OUT/meta.json.tmp" && mv "$OUT/meta.json.tmp" "$OUT/meta.json"

# --- per-file diff, as in prefetch.sh ---
rm -rf "$OUT/diff" "$OUT/diff-index.txt"
mkdir -p "$OUT/diff"
awk -v dir="$OUT/diff" '
  /^diff --git / {
    n++
    path = $4
    sub(/^b\//, "", path)
    flat = path
    gsub(/\//, "__", flat)
    out = sprintf("%s/%03d_%s.diff", dir, n, flat)
    print path "\t" out > (dir "/../diff-index.txt")
  }
  n { print > out }
' "$OUT/full.diff"
touch "$OUT/diff-index.txt"

# --- new definitions at <commit>, as in prefetch.sh ---
: >"$OUT/new-symbols.txt"
while IFS=$'\t' read -r path chunk; do
  case "$path" in *.py) ;; *) continue ;; esac
  names=$(grep -hoE '^\+[[:space:]]*(async def|def|class)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' \
    "$chunk" | awk '{print $NF}' | sort -u || true)
  for name in $names; do
    git grep -nE "^[[:space:]]*(async def|def|class)[[:space:]]+${name}[[:space:]]*[(:]" \
      "$AT" -- "$path" 2>/dev/null | sed "s|^$AT:|  |" >>"$OUT/new-symbols.txt" || true
  done
done <"$OUT/diff-index.txt"
sort -u -o "$OUT/new-symbols.txt" "$OUT/new-symbols.txt"

grep -hE '^\+[[:space:]]*(async )?def test_|^\+class Test' "$OUT/full.diff" \
  | sed -E 's/^\+[[:space:]]*//' >"$OUT/new-tests.txt" || true

echo "PR $PR at ${AT:0:10} (base ${BASE:0:10}): $NFILES files, +$ADD/-$DEL, $(wc -l <"$OUT/commits.txt" | tr -d ' ') commits"
