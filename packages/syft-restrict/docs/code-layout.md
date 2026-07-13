# Code layout

Map of `src/syft_restrict/`:

| Module          | Role                                                                  |
| --------------- | --------------------------------------------------------------------- |
| `astutil.py`    | Line ranges, import scan, small syntax helpers. No policy.            |
| `policy.py`     | Allow lists, safe/banned builtins, decorator/hook lists, `Policy`.    |
| `verifier.py`   | Static checker: walks private lines default-deny, reports violations. |
| `obfuscator.py` | After a clean verify: rename private identifiers, blank constants.    |
| `runner.py`     | `run()` — verify → obfuscate → certificate.                           |
| `errors.py`     | `RestrictError`, `PolicyViolation`.                                   |
| `__init__.py`   | Public exports.                                                       |

Tests under `tests/verify/`:

| File                      | Role                                                            |
| ------------------------- | --------------------------------------------------------------- |
| `test_whitelist.py`       | Green path (aligns with [verify.md](verify.md))                 |
| `test_whitelisted_lib.py` | Library paths + operator bundles                                |
| `test_disallowed.py`      | Default-deny catalog (aligns with [blacklist.md](blacklist.md)) |
| `test_bypasses.py`        | Multi-step escape regressions                                   |
| `test_ranges.py`          | Invalid private ranges                                          |
