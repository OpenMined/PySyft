# Gemma enclave benchmark prompts

`safety_prompts_mock.csv` (mock) and `safety_prompts.csv` (private) are a deterministic 10-row sample (first prompt per hazard) of the MLCommons AILuminate demo prompt set — <https://github.com/mlcommons/ailuminate/blob/main/airr_official_1.0_demo_en_us_prompt_set_release.csv> — with columns renamed/reordered to match the real AILuminate reserve prompt set: `release_prompt_id` → `prompt_uid`, `persona` and `prompt_hash` dropped, order `prompt_uid, hazard, locale, prompt_text`.

Used by `colab/2. DO-benchmark-owner-gemma-restrict.ipynb` (fetched over HTTPS) and `dev/1. enclave_gemma_inmem_restrict_ailumniate.ipynb` (read from this checkout).
