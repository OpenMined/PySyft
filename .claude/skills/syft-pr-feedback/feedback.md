# Feedback archive

Every piece of feedback on `syft-pr-feedback`, its rules and its invariants. Only append to this
file. Nothing is deleted, even after a rule covers it. Read the whole file before you propose a rule,
so you notice when the same complaint comes up again.

## Format

    ## YYYY-MM-DD — PR <N>, short label
    Kind: missed | wrong | bad fix | report | new rule | new invariant
    By: who gave the feedback
    Feedback:
    > what they said, word for word
    Rule: the one-sentence rule taken from it, or "none"
    Landed: the file and the rule id it changed, or "covered by <rule id>", or "proposed, not merged"

---

## 2026-11-13 — first version

Kind: new rule, new invariant
By: mined from review comments, and the team meeting on review guidelines

Feedback:

> Koen: "it would be great if you would have a skill that kind of catches the same type of problems
> that I find ... if you have a PR, you call the skill and it acts as if I'm reviewing your PR."
>
> Meeting: favour established terminology ("if it's a queue, use a queue"); a method name says what
> it acts on (`submit_job`, not `submit`), except in user-facing APIs; complexity has to be
> justified by the need for the feature.

Rule: 37 rules, D1–S4, mined from 180 comments by Koen and 77 by Pedro on the last 200 merged pull
requests, and 23 invariants in 7 areas. All are `candidate`.

Landed: `docs/engineering/review-rules.md`, `.claude/rules/invariants/`.
