#!/usr/bin/env bash
#
# Gather everything the syft-pr-review skill reads, in one call.
#
# Usage: .claude/skills/syft-pr-review/prefetch.sh <pr-number> [outdir]
#        outdir defaults to /tmp/pr-<pr-number>
#
# Writes to outdir and prints an index. Read-only with respect to the working
# tree: it fetches refs and reads files out of them with `git show` / `git grep`,
# so it never checks anything out and never needs the PR branch checked out.
#
# Keep this in step with SKILL.md. When the skill asks for something this does
# not gather, add it here rather than running the command by hand.

set -euo pipefail

PR="${1:?usage: prefetch.sh <pr-number> [outdir]}"
OUT="${2:-/tmp/pr-$PR}"

rm -rf "$OUT/diff"
mkdir -p "$OUT/diff"

# ---------------------------------------------------------------- metadata ---
gh pr view "$PR" \
  --json title,body,author,state,baseRefName,headRefName,additions,deletions,changedFiles,url,isDraft,mergeable \
  >"$OUT/meta.json"

jq -r '.body // ""' "$OUT/meta.json" >"$OUT/body.md"

HEAD_REF=$(jq -r .headRefName "$OUT/meta.json")
BASE_REF=$(jq -r .baseRefName "$OUT/meta.json")
DEFAULT_REF=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name)

# ------------------------------------------------------------------- files ---
gh pr view "$PR" --json files \
  -q '.files[] | "\(.additions)+ \(.deletions)- \(.path)"' \
  | sort -rn >"$OUT/files.txt"

# Lock files, notebooks and generated data are counted and skipped, not read.
# `.claude/` too: agent tooling is not the PR author's work, and it rides along
# in the diff whenever a skill edit is landed on the branch before it lands on
# the base. Reviewing it would attribute it to the wrong person.
grep -E '(\.lock|lock\.json|\.ipynb|\.min\.(js|css)|\.svg|\.png|\.csv|\.parquet)$|(^| )\.claude/' \
  "$OUT/files.txt" >"$OUT/files-skippable.txt" || true

# ----------------------------------------------------------------- history ---
git fetch -q origin "$HEAD_REF" "$BASE_REF" "$DEFAULT_REF" 2>/dev/null || true

git log --oneline --no-merges "origin/$HEAD_REF" "^origin/$BASE_REF" >"$OUT/commits.txt"
git log --oneline --merges "origin/$HEAD_REF" "^origin/$BASE_REF" >"$OUT/merges.txt"

# Step 1 of the skill: does the description name a base branch that has not
# landed yet? Every local branch named in the body is tested for ancestry, so
# an unmerged base shows up as NOT-MERGED and its commits land with this PR.
{
  echo "head=$HEAD_REF base=$BASE_REF repo-default=$DEFAULT_REF"
  for ref in $(grep -oE '\b[a-z0-9._-]+/[a-z0-9._/-]+\b' "$OUT/body.md" 2>/dev/null | sort -u); do
    git rev-parse --verify -q "origin/$ref" >/dev/null 2>&1 || continue
    if git merge-base --is-ancestor "origin/$ref" "origin/$DEFAULT_REF"; then
      echo "named-base $ref MERGED into $DEFAULT_REF"
    else
      echo "named-base $ref NOT-MERGED into $DEFAULT_REF — its commits land with this PR"
    fi
  done
} >"$OUT/base-check.txt"

# -------------------------------------------------------------------- diff ---
gh pr diff "$PR" >"$OUT/full.diff"

# One file per changed path, so a chunk is `cat`-ed instead of hunted for with
# `grep -n '^diff --git'` and `sed -n`.
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

# --------------------------------------------------------- new definitions ---
# Every def/class the diff adds, with its real line number on the PR head, so
# `path/file.py:<line>` can be cited without a second pass per symbol.
: >"$OUT/new-symbols.txt"
while IFS=$'\t' read -r path chunk; do
  case "$path" in *.py) ;; *) continue ;; esac
  names=$(grep -hoE '^\+[[:space:]]*(async def|def|class)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' \
    "$chunk" | awk '{print $NF}' | sort -u || true)
  for name in $names; do
    # No \b: git grep's ERE does not support it. A def is followed by "(",
    # a class by "(" or ":", which is boundary enough to not match a prefix.
    git grep -nE "^[[:space:]]*(async def|def|class)[[:space:]]+${name}[[:space:]]*[(:]" \
      "origin/$HEAD_REF" -- "$path" 2>/dev/null \
      | sed "s|^origin/$HEAD_REF:|  |" >>"$OUT/new-symbols.txt" || true
  done
done <"$OUT/diff-index.txt"
sort -u -o "$OUT/new-symbols.txt" "$OUT/new-symbols.txt"

# ------------------------------------------------------------------- tests ---
# Test functions the diff adds, per file, so section 4 gets one bullet each
# without re-reading the diff.
grep -hE '^\+[[:space:]]*(async )?def test_|^\+class Test' "$OUT/full.diff" \
  | sed -E 's/^\+[[:space:]]*//' >"$OUT/new-tests.txt" || true

# ------------------------------------------------------------------- index ---
cat <<SUMMARY
prefetch for PR $PR -> $OUT

  meta.json             $(jq -r '"\(.title)  [\(.state)] \(.author.login)  \(.headRefName) -> \(.baseRefName)  +\(.additions)/-\(.deletions) across \(.changedFiles) files"' "$OUT/meta.json")
  body.md               PR description, to check its claims against the diff
  files.txt             $(wc -l <"$OUT/files.txt" | tr -d ' ') changed files, largest first
  files-skippable.txt   $(wc -l <"$OUT/files-skippable.txt" | tr -d ' ') lock/generated/notebook files — count them, do not read them
  commits.txt           $(wc -l <"$OUT/commits.txt" | tr -d ' ') non-merge commits on $HEAD_REF not in $BASE_REF
  merges.txt            $(wc -l <"$OUT/merges.txt" | tr -d ' ') merge commits in the same range
  base-check.txt        $(sed -n '2,$p' "$OUT/base-check.txt" | wc -l | tr -d ' ') branch(es) named in the body, with ancestry
  full.diff             $(wc -l <"$OUT/full.diff" | tr -d ' ') lines
  diff-index.txt        path -> per-file diff, $(wc -l <"$OUT/diff-index.txt" | tr -d ' ') entries; cat the chunk you need
  new-symbols.txt       $(wc -l <"$OUT/new-symbols.txt" | tr -d ' ') added def/class, with line numbers on $HEAD_REF
  new-tests.txt         $(wc -l <"$OUT/new-tests.txt" | tr -d ' ') added test functions/classes

$(cat "$OUT/base-check.txt")
SUMMARY
