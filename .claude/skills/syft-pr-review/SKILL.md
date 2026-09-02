---
name: syft-pr-review
description: Write a review document for a syft pull request. Takes a PR number or URL, reads the diff, and produces a checkbox-structured summary of the flows that changed, what was added, each individual change, the tests, and any code-standards problems in the new code. Use when the user says "review PR 1234", "write me a review doc for <PR link>", or "what changed in this PR".
allowed-tools: Bash(gh:*), Bash(git:*), Bash(cat:*), Bash(sed:*), Bash(grep:*), Bash(rg:*), Bash(mkdir:*), Bash(wc:*), Bash(ls:*), Bash(jq:*), Bash(bash:*), Bash(./.claude/skills/syft-pr-review/prefetch.sh:*), Read, Write, Glob, Grep
---

# Review a syft PR

Write a document a reader finishes in a few minutes and still knows **what changed** and **what was
decided**. Explain the change, and flag where the new code breaks a house standard. Do not hunt for
bugs — that is `/code-review`.

## Input

A PR number or URL. If none was given, ask for it.

## Step 1 — Prefetch, in one call

```bash
./.claude/skills/syft-pr-review/prefetch.sh <N>       # writes /tmp/pr-<N>/
```

It prints an index and writes:

| file                  | what is in it                                                       |
| --------------------- | ------------------------------------------------------------------- |
| `meta.json`           | title, body, author, state, base, head, +/-, file count             |
| `body.md`             | the description, to check its claims against the diff               |
| `files.txt`           | every changed path with its +/-, largest first                      |
| `files-skippable.txt` | lock files, notebooks and generated data — count these, do not read |
| `commits.txt`         | non-merge commits on the head branch that are not in the base       |
| `merges.txt`          | merge commits in that same range                                    |
| `base-check.txt`      | every branch named in the body, and whether it has landed yet       |
| `full.diff`           | the whole diff                                                      |
| `diff-index.txt`      | `path<TAB>per-file diff`; `cat` the chunk you want                  |
| `new-symbols.txt`     | every `def`/`class` the diff adds, with its line number on the head |
| `new-tests.txt`       | every test function and class the diff adds                         |

Read `base-check.txt` first. A branch marked NOT-MERGED is a base that has not landed, so its
commits land with this PR and belong in the document — say so in one sentence at the top and tag
each item with the branch it came from. `commits.txt` is the author's own work.

`new-symbols.txt` is a candidate list, not the answer: it also catches a nested function and a
pre-existing `def` whose signature changed. Confirm each one is really new before you call it new.

**Keep the script in step with this file.** If a step below needs something `/tmp/pr-<N>/` does not
hold — because the skill changed, or because this PR has a shape the script does not cover — add it
to `prefetch.sh` and re-run, rather than running the command by hand. The point of the script is
that one call answers the whole of steps 1 and 2.

## Step 2 — Read the diff, not the description

`cat` the per-file chunks from `diff-index.txt`, source files first, then the places that call them
— that is where the flows are. For tests, read the name and what it asserts, not the whole body.
Say what you skipped, with counts, using `files-skippable.txt`. Open the changed file from the repo
when a diff chunk alone is not enough to describe something correctly.

Check before you write: whether a name is a method or a plain function, whether a folder is really a
package, and whether each claim in `body.md` is still true.

## Step 3 — Write it

Save to `koen/pr-reviews/pr-<N>-<slug>.md` when a git-ignored `koen/` folder exists, otherwise
`pr-reviews/` in the repo root — and say the file is untracked.

Flows come first, because they are why the reader opened the document.

```md
# PR <N> — <title>

`<repo>#<N>` · <author> · <head> → <base> · +<add>/-<del> across <files> files · <state>

<2-4 plain sentences: what this PR does and why.>

> Read time ~<n> min. Not read: <what, with counts>.

---

- [ ] **1. Flows**

  - [ ] **1.1 <Flow name>** _(new)_
    - [ ] When <trigger>, `A.start()` calls `B.handle()`, which calls `C.save()`.
    - [ ] Decision: we could have done Z; we do A because <reason>.
  - [ ] **1.2 <Flow name>** _(changed)_
    - [ ] When we <do X>, we previously <did Z>. Now <A>.
    - [ ] Now `Class.method()` computes X and calls `other.py: thing()`.

- [ ] **2. What is new** — additions only

  - [ ] **NEW CLASS `Name`** — `path/file.py:<line>` — <what it is for>. Built by `A.b()`. (Flow 1.2)
  - [ ] **NEW MODULE `path/file.py`** — <what it is for>. Defines `file.py: func()`. (Flow 1.1)
  - [ ] **NEW helpers in `path/file.py`** — `f()` <does X>, `g()` <does Y>. (Flow 1.1)

- [ ] **3. Changes** — everything changed or deleted except tests, grouped by theme

  - [ ] **A — <Theme>**
    - [ ] **A1 <label>** (Flow 1.2) — When we <do X>, we previously <did Z>, now we <do A>.
          `path/file.py: Class.method()`
    - [ ] **A2 <label>** — DELETED `Class.old()` and `file.py: helper()`, because <reason>.

- [ ] **4. Tests**

  - [ ] **NEW `path/test_file.py`** — <n> tests<, and how they are set up, when it is worth a clause>
    - [ ] `test_a()` — <the condition, what runs, and what is then true>
    - [ ] `test_b()` — <the condition, what runs, and what is then true>
  - [ ] **REWRITTEN `test_c()`** — `path/test_file.py` — now asserts <X> instead of <Y>, because
        <reason>.
  - [ ] **UPDATED for the new code** — <n> tests across <n> files follow the new
        `Class.method()` signature. Nothing is asserted differently.
  - [ ] Decision: <only when a test settles something a reader would otherwise wonder about>

- [ ] **5. Code standards** — only where the new code breaks one; drop the section when it does not

  - [ ] `path/file.py: Class.method()` — <the problem, one line>

- [ ] **6. Blocked on** — drop this section unless something real stops the merge
  - [ ] <the problem, and what has to happen before this can go in>
```

### What counts as a code-standards problem

Judge only the lines this PR touched. Do not review the code around them, and do not turn this into
a second review. Name the file and the `Class.method()` or `file.py: function()`, keep each to one
line, and write nothing when there is nothing wrong. Look for:

- a function or method you have to scroll to read; aim for 30 lines, and allow more only when it
  calls nothing else. Test functions are exempt — a long test is normal, so never flag one for
  length
- a method that calls out and then works on the answer inline — `ab = Y.Z()` followed by logic on
  `ab` belongs in its own helper
- string building that is not an f-string
- a repeated or magic value that belongs in a module-level constant
- two functions or methods that do substantially the same work; one of them belongs, called from
  both places. Name the pair and what they share. Let a repeated line or two go — the copy has to be
  big enough that pulling it out is worth a helper
- an import inside a function; fine only to break a circular import, and worth one short note
- a name that does not say what the thing is, or pads a name that does. Length is free when every
  word earns it, so drop articles and filler. Say it as a plain sentence and stop there:
  ``test_no_hint_when_the_do_owns_none_of_the_jobs() violates the no-filler rule, rename to
  test_no_hint_when_do_owns_no_jobs()``

## Rules

- [ ] **Every bullet is a checkbox**, at every depth.
- [ ] **Scannable, not prose.** Keep a bullet to two or three lines. When a bullet reaches for a
      semicolon to join items, or repeats the same shape three times over, those items are separate
      child bullets. The reader is looking for one thing, not reading front to back.
- [ ] **Length and position follow the code.** Word count and placement are both claims about how
      much something matters, so a ten-line class gets a few words, not five bullets. Group small
      related additions under one bullet. Collect mechanical cleanups — a moved import, a renamed
      local, a deleted comment — into one short note at the end of the theme they belong to, never a
      bullet of their own.
- [ ] **Say it once.** Section 2 describes code that was added. Section 3 covers what behaviour
      changed and what was deleted, and refers to new code by name rather than describing it again.
      Tests belong only in section 4.
- [ ] **One test, one bullet, nested under its file.** The file gets the parent bullet with the
      count, each test a child bullet. Do not walk through test bodies.
- [ ] **A test bullet carries its own context.** One sentence naming the condition, what runs, and
      what is then true — the reader has only that line. Write _if a job list holds no unique name,
      the hint under the table renders a position instead of a name_, not _with no unique name left,
      the hint gives a position_. Give the minimum that makes it stand up, not the whole setup.
- [ ] **Tests, in proportion.** Collapse tests that only follow the new code — a changed call, a
      rebuilt fixture — into a single bullet with a count, since nothing is asserted differently.
      Give a bullet of its own to a test whose meaning actually changed, and say why.
- [ ] **Standards: new code only.** Flag a standards problem only in lines this PR touched, and
      only when there is one. Never audit the surrounding code.
- [ ] **No question list.** A reviewer can ask their own questions, so do not collect them. Only
      name something that genuinely blocks the merge, and leave section 6 out when nothing does.
- [ ] **Plain words.** Explain any term you have to use. Do not write "on the wire" (say _sent over
      the network_), "opaque", "envelope", "surface", "inert" or "residual".
- [ ] **Every bullet stands alone.** A reader three levels deep must not need the bullet above it. A
      theme heading is a filing label, not context: "B — Errors that name both sides" says nothing
      about what raises the error or who the two sides are, so every bullet under it names its own
      subject.
- [ ] **Name both sides.** Never just "a version mismatch" — say which versions, held by whom, and
      compared against what: this client against the peer's version file, the local folder against the
      copy on Drive, one protocol against another. The same goes for any comparison or hand-off.
- [ ] **Qualify every name.** `Class.method()` for a method, `file.py: function()` for a plain
      function. Never a bare `_helper()` — the reader will guess the wrong owner.
- [ ] **Say who calls whom, and when it runs.** "`A.b()` calls `c.py: d()`", not "`A.b()` → `d()`".
- [ ] **Before and after, everywhere:** _When we do X, we previously did Z. Now we do A._
- [ ] **Say what was decided** — the alternative and why it lost — not only what changed. Only for
      decisions that change how someone reads the code; skip the rest.
- [ ] **Budget reading time:** roughly 40 bullets for a normal PR, 120 for a very large one.
- [ ] Do not repeat the PR description. If your text matches it, you read the wrong thing.
- [ ] Cross-reference by name, e.g. `(Flow 1.2)`. Links such as `[x](#slug)` do not work: GitHub
      gives headings no `id` inside a comment, so there is nothing to point at.

## Finally

Print the file path and a short spoken summary. Do not paste the document back.

Offer to post it as a PR comment. If the user agrees, post a **new** comment — never edit an
existing one, because editing discards any checkboxes they have already ticked. Their ticks can be
recovered from `userContentEdits` in the GraphQL API, but only before the history is trimmed.
