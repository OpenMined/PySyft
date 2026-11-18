## Summary

<!--
One bullet per use case this PR changes: "When we do X, we previously did Y. Now we do Z."
If you change something users see directly, add a bullet that says what changes for them:
"When we call create_dataset, we previously passed X, which caused Y. Now we pass Z, which causes A."
-->

-

## Guidelines

<!--
Rate this PR against dev-guidelines/guidelines.md. Keep one colour per line:
🟢 follows the guideline, 🟠 partly follows it, 🔴 breaks it.
For 🟠 and 🔴, say why in one line and point to the most important problem.
Keep it short.
-->

**High level**

- 🟢 🟠 🔴 Split: <how confident you are that this PR should be split>
- 🟢 🟠 🔴 Size: <this PR in one sentence; if it is too big, the split you suggest>

**Mid level**

- 🟢 🟠 🔴 User-facing code
- 🟢 🟠 🔴 Error handling
- 🟢 🟠 🔴 State versioning
- 🟢 🟠 🔴 Function structure
- 🟢 🟠 🔴 Tests

**Low level**

- 🟢 🟠 🔴 Naming
- 🟢 🟠 🔴 Code style

## Security

<!-- If this PR changes the security properties of the system, summarize how in a few lines. Otherwise write "No change." -->

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
