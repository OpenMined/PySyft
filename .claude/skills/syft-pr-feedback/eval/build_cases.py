"""Build back-test cases from past reviews.

A case is one pull request at one commit that a reviewer commented on, with the
comments the reviewer left there. The skill reviews the pull request at that
commit, and a judge compares its findings with the comments.

Usage:
    python3 .claude/skills/syft-pr-feedback/eval/build_cases.py [PR ...]

Without PR numbers it uses DEFAULT_PRS. Writes cases.json next to this file.
Needs `gh`, logged in.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_OWNER, REPO_NAME = "OpenMined", "PySyft"
REVIEWERS = {"koenvanderveen": "K", "pjwerneck": "P"}
# Merged PRs with at least one substantive review by a reviewer above.
DEFAULT_PRS = [
    9434,
    9436,
    9438,
    9443,
    9487,
    9497,
    9520,
    9522,
    9523,
    9536,
    9561,
    9562,
    9564,
    9571,
    9604,
]
NOT_A_FINDING = {"lgtm", "-", "awesome!", "test2"}

QUERY = """
query($owner: String!, $name: String!, $n: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $n) {
      number title author { login } mergeCommit { oid }
      reviews(first: 60) { nodes { author { login } state body url commit { oid } } }
      reviewThreads(first: 100) {
        nodes {
          isResolved isOutdated path originalLine line
          comments(first: 40) { nodes { author { login } body url originalCommit { oid } } }
        }
      }
    }
  }
}
"""


def fetch(pr: int) -> dict:
    result = subprocess.run(
        [
            "gh",
            "api",
            "graphql",
            "-F",
            f"owner={REPO_OWNER}",
            "-F",
            f"name={REPO_NAME}",
            "-F",
            f"n={pr}",
            "-f",
            f"query={QUERY}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["data"]["repository"]["pullRequest"]


def login(node: dict) -> str | None:
    return (node.get("author") or {}).get("login")


def build_case_items(pr: dict) -> dict[str, list[dict]]:
    """Maps each reviewed commit to the review items left on it."""
    by_commit: dict[str, list[dict]] = defaultdict(list)
    for thread in pr["reviewThreads"]["nodes"]:
        comments = thread["comments"]["nodes"]
        first = comments[0] if comments else None
        if not first or login(first) not in REVIEWERS:
            continue
        commit = (first.get("originalCommit") or {}).get("oid")
        if not commit:
            continue
        by_commit[commit].append(
            {
                "reviewer": REVIEWERS[login(first)],
                "path": thread["path"],
                "line": thread["originalLine"] or thread["line"],
                "body": first["body"].strip(),
                "addressed": thread["isResolved"] or thread["isOutdated"],
                "url": first["url"],
            }
        )
    for review in pr["reviews"]["nodes"]:
        body = (review["body"] or "").strip()
        commit = (review.get("commit") or {}).get("oid")
        if login(review) not in REVIEWERS or not commit or not body:
            continue
        if body.lower() in NOT_A_FINDING:
            continue
        by_commit[commit].append(
            {
                "reviewer": REVIEWERS[login(review)],
                "path": None,
                "line": None,
                "body": body,
                "addressed": None,
                "url": review["url"],
            }
        )
    return by_commit


def main() -> None:
    prs = [int(a) for a in sys.argv[1:]] or DEFAULT_PRS
    cases = []
    for number in prs:
        pr = fetch(number)
        for commit, items in sorted(build_case_items(pr).items()):
            for i, item in enumerate(items, start=1):
                item["id"] = f"{number}@{commit[:7]}#{i}"
            cases.append(
                {
                    "pr": number,
                    "title": pr["title"],
                    "author": login(pr),
                    "commit": commit,
                    "reviewers": sorted({item["reviewer"] for item in items}),
                    "items": items,
                }
            )
    out = Path(__file__).with_name("cases.json")
    out.write_text(json.dumps(cases, indent=2) + "\n")
    n_items = sum(len(c["items"]) for c in cases)
    print(f"{len(cases)} cases, {n_items} review items -> {out}")


if __name__ == "__main__":
    main()
