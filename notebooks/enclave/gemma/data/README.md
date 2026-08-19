# Gemma enclave benchmark prompts

`safety_prompts_mock.csv` (mock) and `safety_prompts.csv` (private) are a deterministic 10-row sample (first prompt per hazard) of the MLCommons AILuminate demo prompt set — <https://github.com/mlcommons/ailuminate/blob/main/airr_official_1.0_demo_en_us_prompt_set_release.csv> — with columns renamed/reordered to match the real AILuminate reserve prompt set: `release_prompt_id` → `prompt_uid`, `persona` and `prompt_hash` dropped, order `prompt_uid, hazard, locale, prompt_text`.

`safety_prompts_clean.csv` and `safety_prompts_mock_clean.csv` are those same rows, re-quoted. Like the reserve set it mimics, the source pair quotes inconsistently: a prompt containing a comma is left unquoted, so `csv` splits it across fields and truncates the prompt, while a prompt without one is quoted for no reason. The `_clean` copies quote only where a comma requires it, so they read correctly with a plain `csv.DictReader`. Row content is identical — only quoting differs.

**The notebooks read the `_clean` copies.** Regenerate them after editing either source file:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

COLUMNS = ["prompt_uid", "hazard", "locale", "prompt_text"]
REST = "__extra__"  # csv collects fields past the header here; for us it is a spilled prompt_text tail

for src, dst in [("safety_prompts.csv", "safety_prompts_clean.csv"),
                 ("safety_prompts_mock.csv", "safety_prompts_mock_clean.csv")]:
    with open(src, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, restkey=REST)
        assert reader.fieldnames == COLUMNS, reader.fieldnames
        rows = []
        for row in reader:
            # prompt_text is the last column, so anything past the header belongs to it, and
            # re-joining with "," restores the original text exactly.
            extra = row.pop(REST, None)
            if extra:
                row["prompt_text"] = ",".join([row["prompt_text"], *extra])
            assert all(row.values()), f"{src}: empty column in {row}"
            rows.append(row)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{src} -> {dst}: {len(rows)} rows")
PY
```

Used by `colab/2. DO-benchmark-owner-gemma-restrict.ipynb` (fetched over HTTPS from the `enclave-mvp` branch) and `dev/1. enclave_gemma_inmem_restrict_ailumniate.ipynb` (read from this checkout).
