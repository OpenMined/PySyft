---
name: syft-pr-review
description: Write a review document for a syft pull request. Takes a PR number or URL, reads the diff, and produces a checkbox-structured summary of the flows that changed, what was added, each individual change, the tests, and any code-standards problems in the new code. Use when the user says "review PR 1234", "write me a review doc for <PR link>", or "what changed in this PR".
allowed-tools: Bash(gh:*), Bash(git:*), Bash(cat:*), Bash(sed:*), Bash(grep:*), Bash(rg:*), Bash(mkdir:*), Bash(wc:*), Bash(ls:*), Read, Write, Glob, Grep
---

# Review a syft PR

Write a document a reader finishes in a few minutes and still knows **what changed** and **what was
decided**. Explain the change, and flag where the new code breaks a house standard. Do not hunt for
bugs — that is `/code-review`.

## Input

A PR number or URL. If none was given, ask for it.

## Step 1 — Load the PR, and find the branch it really builds on

```bash
gh pr view <N> --json title,body,author,state,baseRefName,headRefName,additions,deletions,changedFiles
gh pr view <N> --json files -q '.files[] | "\(.additions)+ \(.deletions)- \(.path)"' | sort -rn
```

If the description names a base branch other than the GitHub one, check whether it is merged yet:
`git merge-base --is-ancestor origin/<named-base> origin/<default>`. If it is not, its commits land
with this PR, so review those too — do not shrink the document to the author's own commits. Say so
in one sentence at the top and tag each item with the branch it came from. The author's own commits:

```bash
git log --oneline --no-merges origin/<head> ^origin/<default> ^origin/<named-base>
```

## Step 2 — Read the diff, not the description

```bash
gh pr diff <N> > /tmp/pr-<N>.diff
grep -n '^diff --git' /tmp/pr-<N>.diff    # file boundaries; read the chunks with sed
```

Read source files first, then the places that call them — that is where the flows are. For tests,
read the name and what it asserts, not the whole body. Count and skip lock files, generated data,
fixtures and reformatting, and say what you skipped. Open the changed file from the repo when a diff
chunk alone is not enough to describe something correctly.

Check before you write: whether a name is a method or a plain function, whether a folder is really a
package, and whether each claim in the PR description is still true.

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

  - [ ] **NEW `path/test_file.py`** — <n> tests: `test_a()` <what it checks>, `test_b()`
        <what it checks>.
  - [ ] **REWRITTEN `test_c()`** — now asserts <X> instead of <Y>, because <reason>.
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

- a function or method you have to scroll to read; aim for 10 lines, and allow more only when it
  calls nothing else
- a method that calls out and then works on the answer inline — `ab = Y.Z()` followed by logic on
  `ab` belongs in its own helper
- string building that is not an f-string
- a repeated or magic value that belongs in a module-level constant
- an import inside a function; fine only to break a circular import, and worth one short note
- a test whose name does not say what it checks; a long name is fine

## Rules

- [ ] **Every bullet is a checkbox**, at every depth.
- [ ] **Length follows the code.** A ten-line class gets a few words, not five bullets. Group small
      related additions under one bullet. Give something its own top-level bullet only when a reader
      needs it on its own.
- [ ] **Say it once.** Section 2 describes code that was added. Section 3 covers what behaviour
      changed and what was deleted, and refers to new code by name rather than describing it again.
      Tests belong only in section 4.
- [ ] **Tests, in proportion.** One sentence per new test, saying what it checks. Collapse tests
      that only follow the new code — a changed call, a rebuilt fixture — into a single bullet with
      a count, since nothing is asserted differently. Give a line to a test whose meaning actually
      changed, and say why. Do not walk through test bodies.
- [ ] **Standards: new code only.** Flag a standards problem only in lines this PR touched, and
      only when there is one. Never audit the surrounding code.
- [ ] **No question list.** A reviewer can ask their own questions, so do not collect them. Only
      name something that genuinely blocks the merge, and leave section 6 out when nothing does.
- [ ] **Plain words.** Explain any term you have to use. Do not write "on the wire" (say _sent over
      the network_), "opaque", "envelope", "surface", "inert" or "residual".
- [ ] **Every bullet stands alone.** A reader three levels deep must not need the bullet above it.
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
