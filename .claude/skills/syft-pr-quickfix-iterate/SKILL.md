---
name: syft-pr-quickfix-iterate
description: Improve the syft-pr-quickfix skill from feedback. Takes a missed finding, a wrong finding, a bad fix, a complaint about the report, or a new check. Logs the feedback to an archive, turns it into a general rule, edits syft-pr-quickfix/SKILL.md within a size budget, and re-runs the check on the PR that prompted the feedback. Use when the user gives feedback on a quickfix report — "finding 3 is wrong", "you missed that X", "also look for Y".
allowed-tools: Bash(gh:*), Bash(git:*), Bash(cat:*), Bash(sed:*), Bash(grep:*), Bash(rg:*), Bash(wc:*), Bash(ls:*), Bash(./.claude/skills/syft-pr-review/prefetch.sh:*), Read, Write, Edit, Glob, Grep, Skill, AskUserQuestion
---

# Improve the syft-pr-quickfix skill

Feedback arrives as one example, such as "finding 3 is not a problem". The job is to find the
general rule behind that example, change the skill, and show the change working on the PR that
prompted the feedback.

There are two files:

- `.claude/skills/syft-pr-quickfix/SKILL.md` is the skill you edit. It has a size budget.
- `.claude/skills/syft-pr-quickfix-iterate/feedback.md` holds every past piece of feedback. Only
  append to it. The archive keeps the detail, so the skill itself can stay short.

Both skills are checked in, through `!` rules in `.gitignore`. Changes to them go to `dev` in a
pull request.

## Input

Feedback in the user's words, usually about a report that was just printed. It is one of five kinds:

- **Missed** — a problem the report should have found.
- **Wrong** — a finding that is not a problem.
- **Bad fix** — a finding that is right, but with a fix that is wrong or too vague.
- **Report** — the report reads badly: too long, unclear, or in the wrong order.
- **New check** — a kind of problem the skill should start looking for.

If no feedback came with the request, ask what should change and which report prompted it.

## Steps

1. Append the feedback to `feedback.md` in the format that file gives. Create the file from that
   format the first time. Then add each finding the user judged to `syft-pr-quickfix/examples.md`,
   under Good or Bad, with the PR, the tag, the place and the reason in one or two lines.
2. Read all of `feedback.md`. Then write the rule as one sentence: _the skill does X, it should do
   Y._ Use **Generalise** below.
3. Decide whether the rule is missing, or whether the skill already says it and was not followed.
   If the skill already says it, rewrite that text so it is clearer or more prominent. Do not add a
   second rule on the same subject.
4. If `~/.claude/skills/write-doc/SKILL.md` exists, read it and follow its rules for every sentence
   you write.
5. Edit `syft-pr-quickfix/SKILL.md`, and stay inside **Size budget** below.
6. Re-run `syft-pr-quickfix` on the PR that prompted the feedback. Show that the missed finding now
   appears, that the wrong finding is gone, or that the report now reads as the user asked.
7. Report back in a few lines: the rule, where it went, the size before and after, and the result
   of the re-run.

## Generalise

Ask what class of problem the example belongs to. Do not fix only the one example.

> "Finding 3 flags a cache as `same-state`" → the rule is "a value that the code rebuilds from
> its source on every change is a cache, not a second source of truth", not "do not flag
> `_peer_cache`".

Test the rule twice before you write it down:

- **Does it apply to code you have not seen?** If it only matches this example, make it broader.
- **Is there enough evidence for it?** One example supports a narrow rule. If you are not sure how
  far the rule goes, ask.

Then read the archive for older feedback that the new rule also covers. When two or three rules are
really one idea, replace them with that one idea, and tell the user which rules you merged.

## Where a rule goes

- A rule about what counts as a problem, or what does not, goes in the text of that check.
- A new kind of problem becomes a new check with its own tag. Before you add one, make sure no
  existing check already covers it.
- A rule about how to confirm a finding goes in **Steps**.
- A rule about how the report reads goes under **Report**. Update the example report as well, so
  that the rules and the example agree.

## Size budget

`syft-pr-quickfix/SKILL.md` stays under **120 lines and 12 checks**. Measure it before and after
the edit:

```bash
wc -l .claude/skills/syft-pr-quickfix/SKILL.md
```

Do not drop a new rule to stay inside the budget. Instead, merge the rule into one that says
something close, replace two narrow rules with one general rule, or tighten the wording of other
rules. Tell the user which of these you did.

## Writing

The skill has to read as plainly as the reports it asks for. Every rule is one short sentence in
everyday words that a reader can test against a report. "Be concise" cannot be tested, so it does
not go in. "At most 15 findings" can be tested.
