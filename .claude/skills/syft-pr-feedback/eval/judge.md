# Judge a back-test run

You compare what the `syft-pr-feedback` skill reported for one pull request at one commit with what
human reviewers wrote at that commit. Run this in a session that did not write the report.

## Input

- One case from `cases.json`: the pull request, the commit, and the review items the reviewers
  left there (`items`).
- The report the skill wrote for that commit: `reviews/<N>@<sha7>/feedback.md`.
- The repository, to check the code at the commit: `git show <commit>:<path>`.

## Steps

1. Mark each review item:
   - `not-a-finding` — praise, approval, a reply, a question about intent with no problem in it,
     or a discussion of out-of-scope work. These do not count.
   - `caught` — a finding names the same place (the same file, and within 15 lines or in the same
     function) and the same problem. A finding that says something is not needed catches a reviewer
     who asked "do we need this?".
   - `partly` — the same place with a related but different problem, or the same problem in
     another place.
   - `missed` — no finding covers it.
2. Mark each finding of the report:
   - `matches` — it caught or partly caught a review item.
   - `extra-valid` — no item covers it, but you checked the code at the commit and it is true and
     worth the author's time.
   - `extra-invalid` — it is wrong, or not worth the author's time.
   - `unsure` — you cannot tell without asking the author.
3. Write the result to `reviews/<N>@<sha7>/judged.json`, then print a three-line summary.

## Output

```json
{
  "pr": 9604,
  "commit": "9466919168...",
  "items": [
    {
      "id": "9604@9466919#1",
      "verdict": "caught",
      "finding": 2,
      "note": "same file, same concern"
    }
  ],
  "findings": [
    { "n": 2, "verdict": "matches", "note": "" },
    {
      "n": 5,
      "verdict": "extra-valid",
      "note": "checked: no caller passes recipients=None"
    }
  ]
}
```

Keep each note to one line. Be strict about `caught`: the place and the problem must both match.
