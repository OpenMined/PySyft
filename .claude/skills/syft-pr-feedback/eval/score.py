"""Score back-test runs judged with judge.md.

Usage:
    python3 .claude/skills/syft-pr-feedback/eval/score.py [reviews-folder]

Reads every reviews/<N>@<sha7>/judged.json and prints, per case and in total:

- recall: (caught + half of partly) / review items that are findings
- recall of addressed items: the same, for items the author acted on
  (the thread was resolved, or the code changed under it)
- precision: (matches + extra-valid) / findings the judge could decide
"""

import json
import sys
from pathlib import Path

CASES = json.loads(Path(__file__).with_name("cases.json").read_text())
ADDRESSED = {item["id"]: item["addressed"] for case in CASES for item in case["items"]}


def recall(items: list[dict]) -> tuple[float | None, int]:
    found = [i for i in items if i["verdict"] != "not-a-finding"]
    if not found:
        return None, 0
    score = sum(
        1 if i["verdict"] == "caught" else 0.5 if i["verdict"] == "partly" else 0
        for i in found
    )
    return score / len(found), len(found)


def precision(findings: list[dict]) -> tuple[float | None, int]:
    decided = [f for f in findings if f["verdict"] != "unsure"]
    if not decided:
        return None, 0
    good = sum(f["verdict"] in ("matches", "extra-valid") for f in decided)
    return good / len(decided), len(decided)


def fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "reviews")
    runs = sorted(folder.glob("*@*/judged.json"))
    if not runs:
        print(f"No judged.json under {folder}/<N>@<sha7>/")
        return
    all_items, all_findings = [], []
    print(
        f"{'case':22} {'recall':>7} {'addressed':>10} {'precision':>10}  items  findings"
    )
    for path in runs:
        run = json.loads(path.read_text())
        items, findings = run["items"], run["findings"]
        addressed = [i for i in items if ADDRESSED.get(i["id"])]
        r, n = recall(items)
        ra, _ = recall(addressed)
        p, m = precision(findings)
        print(
            f"{path.parent.name:22} {fmt(r):>7} {fmt(ra):>10} {fmt(p):>10}  {n:5}  {len(findings):8}"
        )
        all_items += items
        all_findings += findings
    r, n = recall(all_items)
    ra, _ = recall([i for i in all_items if ADDRESSED.get(i["id"])])
    p, _ = precision(all_findings)
    print(
        f"{'total':22} {fmt(r):>7} {fmt(ra):>10} {fmt(p):>10}  {n:5}  {len(all_findings):8}"
    )


if __name__ == "__main__":
    main()
