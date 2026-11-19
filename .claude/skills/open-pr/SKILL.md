---
name: open-pr
description: Open a pull request for the current branch of the syft repo. Runs pre-commit, writes a conventional-commit title, and fills in the repo's PR template. Use when asked to open, create or make a PR, or to push and open a PR.
allowed-tools: Bash(git:*), Bash(gh:*), Bash(pre-commit:*)
---

# Open a PR

1. Check `git status`. If there are uncommitted changes, ask whether to commit them first.
2. Run `pre-commit run --all-files`. When a hook fixes files, commit the fixes. When a hook reports
   a problem it cannot fix, fix it and run pre-commit again, until every hook passes.
3. Push the branch with `git push -u origin <branch>`.
4. Write the title as `type: short description in lowercase`. The allowed types are in the table in
   `.github/PULL_REQUEST_TEMPLATE.md`. The `PR: Lint & Label` check blocks the merge when the title
   does not match.
5. Write the body from `.github/PULL_REQUEST_TEMPLATE.md`, by the rules in
   `dev-guidelines/language.md`. Read `dev-guidelines/guidelines.md` first, because the body rates
   the PR against it.

   - **Summary:** one bullet per use case the PR changes: when we do X, we previously did Y, now we
     do Z. Give each change a user sees its own bullet, and say what changes for the user.
   - **Guidelines:** keep one colour per line: 🟢 follows, 🟠 partly follows, 🔴 breaks. For 🟠
     and 🔴, say why in one line and name the most important problem. Rate honestly; a reviewer
     checks the ratings.
   - **Security:** how the PR changes the security properties, or "No change."
   - **Testing** and **Asana task:** fill them in. If you do not know the Asana task URL, ask.

   Leave out the HTML comments, and the naming convention and auto-labeling notes below the
   sections: they are for the person writing the PR, not for its readers.

6. Write the body to a temporary file and open the PR against `dev`:
   `gh pr create --base dev --title "<title>" --body-file <file>`.
7. Print the PR URL.
