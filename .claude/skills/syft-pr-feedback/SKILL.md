---
name: syft-pr-feedback
description: Give feedback on a syft pull request the way the team's reviewers do. Writes a short summary, then numbered findings against the team's review rules (docs/engineering/review-rules.md) and invariants (.claude/rules/invariants/), each with a concrete fix, so the author can say "fix 2 and 5" or post them as a GitHub review. Takes a PR number or URL. Use right after creating a PR, or when the user says "feedback on PR 1234", "review my PR against our rules", or "what would Koen say about this PR".
argument-hint: <pr-number> [at=<commit>]
allowed-tools: Bash(gh:*), Bash(git:*), Bash(cat:*), Bash(sed:*), Bash(grep:*), Bash(rg:*), Bash(mkdir:*), Bash(wc:*), Bash(ls:*), Bash(jq:*), Bash(python3:*), Bash(./.claude/skills/syft-pr-review/prefetch.sh:*), Bash(./.claude/skills/syft-pr-feedback/context.sh:*), Read, Write, Glob, Grep, Agent
---

# Give feedback on a syft PR

Tell the author, in a few minutes of reading, what their pull request changes and what to fix
before it merges, with each fix concrete enough to apply. Report only what the team's rules,
invariants or a concrete failure support. Saying nothing is a good result for a good pull request.

This skill does not write the full flow document (that is `syft-pr-review`) and does not hunt for
every bug (that is `/code-review`). It reuses the ten checks of `syft-pr-quickfix`.

## Input

A PR number or URL. If none was given, ask for it.

`at=<commit>` reviews the pull request as it was at that commit. It is for back-tests
(`eval/README.md`): the review must then see only what a reviewer saw at that commit.

## Step 1 — Fetch, in two calls

```bash
./.claude/skills/syft-pr-review/prefetch.sh <N>                 # writes /tmp/pr-<N>/
./.claude/skills/syft-pr-feedback/context.sh <N> [--at <commit>] # writes /tmp/pr-<N>/context/
```

`prefetch.sh` is described in `.claude/skills/syft-pr-review/SKILL.md`. `context.sh` adds:

| file                           | what is in it                                            |
| ------------------------------ | -------------------------------------------------------- |
| `context/review-rules.md`      | the rules, read from the base branch                     |
| `context/invariants.md`        | the invariant files whose `paths` match a changed file   |
| `context/existing-comments.md` | comments already on the pull request; do not repeat them |
| `context/head.txt`             | the commit the review is about                           |

With `--at`, `context.sh` first rewrites the diff, file list, commits and new symbols to that
commit, and writes no comments. Then do not read the pull request's comments, reviews or later
commits in any other way either: they hold the answers a back-test checks against.

## Step 2 — Understand before you judge

Read `body.md`, then the per-file chunks in `diff-index.txt`: source files first, then the code that
calls them. Skip the files in `files-skippable.txt`. Open a whole file at the reviewed commit
(`git show <head>:<path>`) when a chunk is not enough. If the body links an issue, read it with
`gh issue view`.

Write the **summary**: three to five plain sentences on what changes, for whom, and why. Then one
line per flow that changed: _when X happens, the code previously did Z; now it does A._ Every later
finding refers to this understanding, so get it right first.

## Step 3 — Find candidates, in three passes

Each pass reads the summary, `context/review-rules.md`, `context/invariants.md` and the diff, and
returns candidates in the format of `passes.md`, next to this file. `passes.md` lists the rules and
questions of each pass:

- **A — Design:** complexity, sources of truth, layers, users, compatibility, scope.
- **B — Correctness and safety:** errors, private attributes, contracts, atomic writes, invariants,
  and concrete bugs.
- **C — Readability:** names, tests, docs, and the ten checks of `syft-pr-quickfix`.

When the diff changes more than 300 lines of source code (not tests, docs or skippable files), run
the three passes as parallel subagents with the Agent tool, `general-purpose`, one per pass. Give
each the path `/tmp/pr-<N>/`, the summary, and its section of `passes.md`, and tell it to read only
and to return candidates only. Otherwise, or when the Agent tool is not available, run the passes
yourself, one after the other.

## Step 4 — Verify every candidate yourself

A pass proposes; you decide. For each candidate:

1. Open the code at the reviewed commit and confirm the claim. Grep the callers before you say a
   case never happens. Grep the readers before you say two places hold the same fact. Name the
   input or the order of events before you say something fails.
2. Drop it when you cannot confirm it, when it is on the "Do not report" list in
   `context/review-rules.md`, when `context/existing-comments.md` already makes the point, or when
   the code it is about was not added or changed by the pull request.
3. Merge candidates with one cause into one finding that lists every place.
4. Set the severity from the "Severity" section of the rules: an invariant broken or a concrete
   failure is 🔴; a rule broken with a concrete fix is 🟡; names, comments and wording are ⚪; a real
   problem outside the scope is ➜. Never raise severity because a pass felt sure.
5. When you are unsure whether a finding is worth reporting, compare it with `examples.md`.

## Step 5 — Write the report

Save it to `reviews/<N>/feedback.md`, or `reviews/<N>@<sha7>/feedback.md` with `at=`. Say the file
is untracked. Then print it in full.

```md
# PR <N> feedback — <title>

<author> · <head> → <base> · +<add>/-<del> in <files> files · reviewed at <sha7>

<Three to five sentences: what changes, for whom, and why.>

- When <trigger>, <the code> previously <did Z>. Now it <does A>.

**Verdict:** <one line, such as "2 must-fix and 3 should-fix; the design is otherwise sound".>

## 🔴 Must fix

1. `JOB-4` · [client.py:412](permalink) `JobClient.list_jobs()`
   `list_jobs()` opens `config.yaml` itself when a job's protocol is "1".
   A job written by the next protocol is then read with the old layout, and fails or shows wrong
   fields; `JobStorage` is meant to be the only reader of job files.
   Fix: call `JobStorage.read_submission(ref)`, and delete the protocol branch.

## 🟡 Should fix

2. `E2` · [settings.py:57](permalink) `file.py: load_settings()`
   The `except Exception` returns the default settings.
   A settings file that fails to parse then silently switches the user to the defaults.
   Fix: catch `json.JSONDecodeError` only, and raise with the path in the message.

## ⚪ Nits

3. `N1` · [manager.py:88](permalink) `DatasetManager.share()` — rename to `share_dataset()`.

## ➜ Follow-ups

- **Cache the peer list per sync** — every call to `peers` reads `SYFT_peers.json` from Drive.

<Not read: what, with counts.> Rules and invariants from `<base>` at `<sha7>`.
```

- Number the findings 1, 2, 3 across all sections, and keep the numbers stable: the author refers to
  them by number. Follow-ups are not numbered.
- Each finding has three parts, in one or two lines each: what the code does, what is wrong with the
  concrete case, and the fix. Name both sides of every comparison.
- Put a ```suggestion block under a finding only when the fix is a few lines and complete.
- Link the place with a permalink: `https://github.com/<owner>/<repo>/blob/<head>/<path>#L<line>`.
  Name it in full: `Class.method()` for a method, `file.py: function()` for a plain function.
- Most impact first within a section. At most 15 findings, of which at most 5 nits. Say how many
  you left out.
- Plain words. Do not write "on the wire", "opaque", "envelope", "surface", "inert" or "residual".
- When nothing survives step 4, write the summary and one line that says there is nothing to fix.

## Finally

Offer three things, and do each only when the user asks:

- **"fix 2 5"** — apply those findings, run the unit tests of each package you touched (`just
test-unit-<package>` or `uv run pytest <path>`), and show the diff. Do not commit.
- **"post"** — post the report as one GitHub review with `event: COMMENT`; never approve and never
  request changes. Each finding whose line is inside a hunk of the diff becomes an inline comment
  (`path`, `line`, `side: RIGHT`, plus `start_line` for a range); the summary, the verdict, the other
  findings and the follow-ups go in the review body. Write the payload to
  `reviews/<N>/review.json`, show it, and post it only after the user confirms:
  `gh api repos/{owner}/{repo}/pulls/<N>/reviews --input reviews/<N>/review.json`.
- **"3 is wrong because …"**, **"you missed …"** — append the feedback to `feedback.md` in the
  format it gives, and add the judged finding to `examples.md`. If the feedback is a new rule or
  changes one, propose the edit to `docs/engineering/review-rules.md` as a `candidate`, and show it.
  Never mark a rule `active` yourself.
