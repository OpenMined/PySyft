---
name: syft-pr-quickfix
description: Check a syft pull request for ten design problems (code paths for cases that never happen, duplicated state, old concepts under new names, half-built objects, the same job done in two places, returning False/None instead of raising, names that do not match the code, vague names, inline values with no name, and nested dict types that do not say what maps to what) and print a short numbered list of easy fixes, with harder findings in an appendix, so the user can say "fix 2 and 5". Takes a PR number or URL. Use when the user says "quickfix PR 1234", "check this PR for quick fixes", or "what should be cleaned up in <PR link>".
allowed-tools: Bash(gh:*), Bash(git:*), Bash(cat:*), Bash(sed:*), Bash(grep:*), Bash(rg:*), Bash(mkdir:*), Bash(wc:*), Bash(ls:*), Bash(jq:*), Bash(./.claude/skills/syft-pr-review/prefetch.sh:*), Read, Write, Glob, Grep
---

# Check a syft PR for quick fixes

Read the code a PR adds or changes and find the ten problems listed under **Checks**. Print
a short numbered list, so the user can ask to fix single findings by their number. This skill does
not hunt for bugs (that is `/code-review`) and does not summarise the PR (that is `syft-pr-review`).

## Input

A PR number or URL. If none was given, ask for it.

## Steps

1. Fetch the PR with the prefetch script of the `syft-pr-review` skill. It writes the diff, split
   per file, to `/tmp/pr-<N>/`:

   ```bash
   ./.claude/skills/syft-pr-review/prefetch.sh <N>
   ```

2. Read the source files in `diff-index.txt`. Skip the files in `files-skippable.txt` and the
   tests, unless a test shows that a case really happens.
3. Go through the checks below for each file. Only flag lines the PR added or changed.
4. Confirm each finding in the code before you write it down. For example, grep the callers before
   you say a case never happens, and grep the readers before you say two fields hold the same
   state. Drop a finding you cannot confirm.
5. When you are unsure whether a finding is worth reporting, compare it with the good and bad
   findings in `examples.md`, next to this file.
6. Write the report, save it, and print it.

## Checks

Each check has a short tag. The tag goes in front of every finding.

1. **`unused-case`** — the code handles a case that never happens or is not worth supporting, such
   as a parameter no caller passes, and that case makes the code harder to read.
2. **`same-state`** — two variables or fields hold the same fact, so they can drift apart. A cache
   that the code keeps in step on purpose is fine.
3. **`old-concept`** — a new name for something that already has a name in the codebase.
4. **`half-built`** — the code creates an object with missing values, then fills them in later. It
   should gather every value first. Report this only when the fix is easy, as defined below.
5. **`done-twice`** — two places do the same job, so a change to that job has to be made twice.
   Report this only when the job takes more than one line and the fix is easy, as defined below.
   A repeated one-liner is not a finding.
6. **`silent-fail`** — a function returns `False` or `None` on failure, where it should raise.
7. **`name-mismatch`** — a name says one thing and the code does another. For example,
   `upload_versioninfo()` takes a `PeerInfo` object.
8. **`vague-name`** — a name does not say what kind of thing it holds, takes or returns. This
   includes a bare verb such as `deliver()`, and a generic word such as `state`, `data` or `info`
   that does not say whose: `_create_current_state_event()` should say file state.
9. **`unnamed-value`** — a value built inline inside another call, so the reader cannot tell what
   it is, such as `files.update(dict.fromkeys(unsent, UNSENT_HASH))`. Fix: a named variable. A call
   whose name says what it returns, such as `self._accepted_changes(...)`, is fine inline.
10. **`unlabelled-mapping`** — a nested type such as `dict[str, dict[str, str]]` does not say what
    its keys and values are. Fix: say it in the docstring, and add a one-line comment such as
    `# peer email -> {path: hash}` where the value is built or declared.

## Easy fixes

The user wants to apply the main list without reviewing each fix closely. So a fix is easy only
when all of these hold:

- It changes one function or a few lines, plus the places that call or read what it renames or
  deletes.
- It needs no design choice. There is one obvious way to do it.
- It keeps the behaviour the PR wants. It deletes, renames, merges or raises, and nothing more.

A finding whose fix is not easy goes to the appendix of the report, not to the main list.

## Report

Save the report to `koen/reviews/<N>/quickfix.md` when the git-ignored `koen/` folder exists.
Otherwise save it to `reviews/<N>/quickfix.md`, and tell the user that the file is untracked. Then
print the report in full, because it is short.

```md
# PR <N> quick fixes — <title>

1. `same-state` · [file.py:<line>](cursor://file/abs/repo/path/file.py:<line>) `Class.method()`
   `self.peer_version` and `self.version_info.version` both hold the peer's version.
   Fix: drop `peer_version` and read `version_info.version`.
2. `silent-fail` · [file.py:<line>](cursor://file/abs/repo/path/file.py:<line>) `file.py: load_peer()`
   `load_peer()` returns `None` when the peer file is missing, and all three callers
   then fail later on `None.email`.
   Fix: raise `PeerNotFoundError` in `load_peer()`.

## Appendix — needs a closer look

A1. `unused-case` · [file.py:<line>](cursor://file/abs/repo/path/file.py:<line>) `Syncer.push()`
The `batch=` branch has no caller, but removing it changes how `push()` splits work.
Fix: drop `batch=` and send one message per change.
```

- Number the findings 1, 2, 3, and keep the numbers stable: the user refers to them by number.
- Put the finding with the most impact first. Put fixes that only change a docstring or comment
  after the fixes that change code.
- Each finding has three parts: the tag and the place, what is wrong, and the fix. Keep each part
  to one or two lines.
- Make the place a link that opens Cursor at the line: `[file.py:<line>](cursor://file<absolute
path>:<line>)`. Then name it in full: `Class.method()` for a method, `file.py: function()` for a plain function.
- Say the problem in plain words. Name both things when two things are involved: both fields, both
  places, or the name and what the code really takes.
- Make the fix concrete enough to apply without asking: name what to delete, rename, merge or raise.
- Put only easy fixes in the main list. Put every other finding under `## Appendix — needs a
closer look`, numbered A1, A2, A3, and say in the finding why the fix is not easy.
- Give at most 15 findings in the main list and 5 in the appendix. When there are more, keep the
  ones with the most impact, and say how many you left out.
- When nothing is wrong, print one line that says so.

## Finally

Offer to apply findings by number, for example "fix 1 and 3". If the user disagrees with a
finding, or names a problem the report missed, offer to feed that back with the
`syft-pr-quickfix-iterate` skill.
