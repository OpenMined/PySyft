# SyftBox Permissions Guide

## The basics

You control who can access files in your datasite by placing a **`syft.pub.yaml`** file in any directory. The datasite owner always has full access to their own datasite -- you cannot lock yourself out.

When your datasite is created, SyftBox sets up your root directory as private and your public folder as readable by everyone. Everything else starts private until you add permission files.

## 1. Simple permissions

Place a `syft.pub.yaml` in a directory. It controls access to all files in that directory and its subdirectories.

```yaml
terminal: false

rules:
  - pattern: '**'
    access:
      admin: []
      write: []
      read: ['*']
```

- **`pattern`** -- which files the rule applies to (`"**"` means everything)
- **`access`** -- who gets what:
  - **`read`**: can read files
  - **`write`**: can read and create/modify files
  - **`admin`**: full control, including changing the permission file itself
- **`terminal`** -- controls whether subdirectories can have their own permission files (explained in section 4)

Values in the access lists are email addresses. Two special values:

- `*` -- everyone (public)
- `*@company.com` -- everyone at a domain

**Make a folder publicly readable:**

```yaml
rules:
  - pattern: '**'
    access:
      read: ['*']
      write: []
      admin: []
```

**Let your company read and write, one person as admin:**

```yaml
rules:
  - pattern: '**'
    access:
      read: ['*@company.com']
      write: ['*@company.com']
      admin: ['alice@company.com']
```

## 2. Patterns

You can have multiple rules in one file with different patterns to give different permissions to different file types.

**Share CSVs with specific people, keep everything else private:**

```yaml
rules:
  - pattern: '**/*.csv'
    access:
      read: ['alice@example.com', 'bob@example.com']
      write: []
      admin: []

  - pattern: '**'
    access:
      read: []
      write: []
      admin: []
```

When a file matches more than one pattern, **the most specific pattern wins**:

- Longer, more precise patterns beat shorter ones
- Exact paths (`reports/q1.csv`) beat wildcards (`**/*.csv`)
- `"**"` is always the lowest priority -- it's your catch-all fallback

So in the example above, a `.csv` file matches both `**/*.csv` and `**`. The CSV rule is more specific, so Alice and Bob can read it. A `.txt` file only matches `**`, so nobody can read it.

Common patterns:

| Pattern                 | Matches                                            |
| ----------------------- | -------------------------------------------------- |
| `"**"`                  | Everything (catch-all)                             |
| `"*.csv"`               | CSV files in this directory                        |
| `"**/*.csv"`            | CSV files in this directory and all subdirectories |
| `"reports/**"`          | Everything inside `reports/`                       |
| `"reports/2024/q1.csv"` | That one specific file                             |

## 3. The `USER` token

There is a special `USER` token you can use in access lists. It means "whoever is currently requesting access." This only makes sense when combined with a **template pattern** that includes the user's identity in the path.

**Example: give each user access to their own folder**

Imagine you have a directory structure like:

```
shared/
  alice@example.com/
  bob@example.com/
```

You can write one rule that gives each user access to only their own folder:

```yaml
rules:
  - pattern: '{{.UserEmail}}/**'
    access:
      read: ['USER']
      write: ['USER']
      admin: []
```

When Alice requests `shared/alice@example.com/file.txt`, the template `{{.UserEmail}}` resolves to `alice@example.com`, the path matches, and `USER` resolves to Alice. So she gets access to her folder but not Bob's.

Available template variables: `{{.UserEmail}}`, `{{.UserHash}}`, `{{.Year}}`, `{{.Month}}`, `{{.Date}}`.

**Important:** if you use `USER` without a template pattern, it just means "any authenticated user" -- the same as `*`.

## 4. Multiple permission files across directories

You can place `syft.pub.yaml` files at different levels of your directory tree. When someone accesses a file, the system walks from the root toward the file and picks **the single closest `syft.pub.yaml`**. Only that file's rules are used.

**There is no merging.** The closest file fully replaces any parent permission files. The parent's rules are completely ignored.

### Example

```
~/Datasite/
  syft.pub.yaml              # (A)
  projects/
    syft.pub.yaml             # (B)
    reports/
      syft.pub.yaml           # (C)
      q1.csv
      readme.txt
    notes/
      todo.txt
```

**File (A)** at the root -- everything private:

```yaml
rules:
  - pattern: '**'
    access:
      read: []
      write: []
      admin: []
```

**File (B)** in `projects/` -- company can read:

```yaml
rules:
  - pattern: '**'
    access:
      read: ['*@company.com']
      write: []
      admin: []
```

**File (C)** in `projects/reports/` -- only Alice can read CSVs:

```yaml
rules:
  - pattern: '**/*.csv'
    access:
      read: ['alice@example.com']
      write: []
      admin: []

  - pattern: '**'
    access:
      read: []
      write: []
      admin: []
```

Now let's trace what happens for each file:

**Accessing `projects/reports/q1.csv`:**
The closest permission file is **(C)**. The system checks (C)'s rules and finds that `**/*.csv` matches -- Alice can read. Note that the company-wide access from **(B)** does **not** apply here. File (B) is ignored because (C) is closer.

**Accessing `projects/reports/readme.txt`:**
The closest permission file is still **(C)**. The `**/*.csv` pattern doesn't match a `.txt` file. The `**` catch-all does match, and it says `read: []`. So nobody can read it. Again, (B)'s company-wide access does **not** fill in -- (C) is in charge and (C) says no.

**Accessing `projects/notes/todo.txt`:**
There is no permission file inside `notes/`. The system walks up and finds **(B)** as the closest. Anyone at `@company.com` can read.

**Accessing a file at the root:**
Only **(A)** exists on the path. Nobody can read (everything is private).

### What if no rule matches?

If the closest permission file has rules but none of them match the file being accessed, **access is denied**. The system does not fall back to a parent permission file.

### The `terminal` flag

Setting `terminal: true` on a permission file **prevents subdirectories from overriding it**. The system stops walking deeper and won't look at any `syft.pub.yaml` files further down the tree.

```yaml
terminal: true

rules:
  - pattern: '**'
    access:
      read: []
      write: []
      admin: []
```

In the example above: if file **(B)** had `terminal: true`, then file **(C)** would be completely ignored. Everything under `projects/` -- including `projects/reports/q1.csv` -- would be governed by (B)'s rules only.

This is useful when you want to guarantee a folder stays locked down and no nested permission file can accidentally open it up.
