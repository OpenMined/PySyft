## Summary

<!-- What does this PR do and why? One or two sentences. -->

## Changes

<!-- Key changes, bulleted. -->

-

## Testing

<!-- How did you verify this works? -->

-

## Asana task

<!-- Paste the Asana task URL for this work. -->

- ***

### PR naming convention

Your PR title **must** follow this format or merging will be blocked:

```
type: short description in lowercase
```

**Example titles:**

- `feat: add retry logic to job approval`
- `fix: handle timeout in notification sender`
- `docs: update syft-bg README`
- `chore: bump dependencies`
- `refactor: split init flow into helpers`
- `test: add criteria validation tests`
- `ci: add release train workflow`

**Allowed types:**

| Type       | When to use                             | Example                             |
| ---------- | --------------------------------------- | ----------------------------------- |
| `feat`     | New feature or capability               | `feat: add DS rejection emails`     |
| `fix`      | Bug fix                                 | `fix: handle empty peer list`       |
| `docs`     | Documentation only                      | `docs: update syft-bg README`       |
| `chore`    | Maintenance, deps, config               | `chore: bump dependencies`          |
| `refactor` | Code restructuring (no behavior change) | `refactor: split init flow`         |
| `test`     | Adding or updating tests                | `test: add approval criteria tests` |
| `ci`       | CI/CD workflow changes                  | `ci: add release train workflow`    |
| `perf`     | Performance improvement                 | `perf: cache Drive API responses`   |
| `build`    | Build system or dependency changes      | `build: pin syft-bg>=0.2.0`         |

Just edit the PR title to fix any errors — the check re-runs automatically.

### Auto-labeling

Labels are applied automatically — you don't need to add them manually:

- **Type labels** from your PR title (e.g., `feat:` adds `feature`, `fix:` adds `bugfix`)
- **Package labels** from which files you changed (e.g., editing `packages/syft-bg/` adds `pkg:syft-bg`)

These labels are used to auto-generate categorized release notes.
