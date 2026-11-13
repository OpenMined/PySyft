# Engineering standards

Three things decide whether a change is good here. Each lives in one place, and people, coding
agents and the review skill all read the same files.

| What                                                                | Where                                | Who reads it                                                                                                                           |
| ------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Invariants**: facts about the system that every change keeps true | `.claude/rules/invariants/<area>.md` | Claude Code loads a file when it reads or edits a file in that file's `paths`; the review skill checks the ones a pull request touches |
| **Review rules**: how we write code                                 | `docs/engineering/review-rules.md`   | reviewers and the review skill; coding agents when you import it (below)                                                               |
| **The review procedure**                                            | `.claude/skills/syft-pr-feedback/`   | Claude Code, when you run `/syft-pr-feedback <PR>`                                                                                     |

## Why invariants are not in a CLAUDE.md

`CLAUDE.md` is git-ignored here, because people keep personal notes in it. Git treats ignored files
as expendable, so a checked-in `CLAUDE.md` could overwrite someone's own file on a pull. Path-scoped
rules in `.claude/rules/` avoid that, and load only for the code they are about.

## Use the review rules while coding

Claude Code does not load `review-rules.md` on its own, because it is long and most sessions do not
need all of it. To have your sessions follow it, add one line to your `CLAUDE.local.md` in the
repository root (git-ignored):

```
@docs/engineering/review-rules.md
```

## Run the review after you open a pull request

`/syft-pr-feedback <PR>` writes a short summary and numbered findings with fixes to
`reviews/<PR>/feedback.md`. To run it each time Claude Code opens a pull request for you, add this to
`.claude/settings.local.json` (personal, git-ignored):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(gh pr create *)",
            "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/skills/syft-pr-feedback/after_pr_create.py"
          }
        ]
      }
    ]
  }
}
```

## Changing a standard

Change the file in a pull request and say why. A rule or invariant starts as `candidate`, and
becomes `active` when the team agrees. The review skill reads rules and invariants from the base
branch, so a pull request cannot change the standards of its own review.
