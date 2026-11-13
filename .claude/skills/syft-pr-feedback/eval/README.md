# Back-testing syft-pr-feedback

Does the skill find what our reviewers find? A back-test answers this on past pull requests. The
skill reviews each one at the commit a reviewer commented on, and a judge compares the findings
with the reviewer's comments.

## Files

| file             | what it does                                                                                 |
| ---------------- | -------------------------------------------------------------------------------------------- |
| `cases.json`     | the cases: pull request, reviewed commit, and the reviewers' comments there                  |
| `build_cases.py` | rebuilds `cases.json` from GitHub                                                            |
| `at_commit.sh`   | makes `/tmp/pr-<N>/` show the pull request at the reviewed commit; `context.sh --at` runs it |
| `judge.md`       | how a judge marks each comment and each finding                                              |
| `score.py`       | turns the judged runs into recall and precision                                              |

## Run one case

1. Pick a case from `cases.json`, and note its `pr` and `commit`.
2. In a fresh session, run `/syft-pr-feedback <pr> at=<commit>`. It writes
   `reviews/<pr>@<sha7>/feedback.md`. The session must not have seen the reviewers' comments.
3. In another fresh session, judge it: give the agent `judge.md`, the case, and the report. It
   writes `reviews/<pr>@<sha7>/judged.json`.
4. `python3 .claude/skills/syft-pr-feedback/eval/score.py reviews`

## Read the numbers with care

- **Recall** is the share of reviewer findings the skill caught. **Recall of addressed items**
  counts only comments the author acted on, which is the better measure. **Precision** is the
  share of the skill's findings that were right and worth fixing.
- Published benchmarks find that models catch 15–31% of what human reviewers flag, and that only
  about half of human review comments lead to a change. A comment the skill misses is not always
  a defect of the skill, and an extra finding is not always noise; the judge checks the code.
- The rules were mined from these same comments, and `review-rules.md` quotes them. A case whose PR
  is quoted in the rules is in-sample: it shows whether the skill applies the rules, not whether
  the rules generalise. Cases that no rule quotes (PR 9522, 9523, 9536) and new pull requests are
  the real test.
- The back-test reads the pull request's final description, which can mention later fixes.
