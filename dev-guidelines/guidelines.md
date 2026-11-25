# Development guidelines

These guidelines apply to all code in this repo. How we write about the code is in `language.md`.
Claude Code reads this file through `CLAUDE.md`. Every PR description rates the PR against the
high-level, mid-level and low-level sections below, and so does the `syft-pr-review` skill.

## High level

- **The right classes and functions.** Use the existing classes when you can. When you add a new
  one, explain why the existing ones do not fit.
- **PR size.** A PR can be described in one sentence. A big PR is not always a problem, but its
  changes must belong together.
- **Justified complexity.** When a feature adds a lot of complexity, such as extra logic in many
  code paths, the need for the feature must justify that complexity.

## Mid level

### User-facing code

- Make what users often do easy, and keep what they sometimes do possible. For the common case, a
  user should not need the docs: autocomplete and the docstring should be enough. For example,
  `share_outputs_with_submitter` is a simple boolean for the common case, and a dict can share the
  outputs with any party.
- Files and folders are a primitive here. They are storage, but also UX, because users sometimes
  work with them directly. So keep the storage format simple in the parts of the syftbox that users
  see.

### Function and method structure

- Keep the condition for doing something in the caller, not inside the function. When the function
  really must check the condition itself, prefix its name with `maybe_`, as in `maybe_do_x()`.

  This also covers a function that returns a value or None, where None means "skip this". Ask the
  question with a separate function instead:

  ```python
  # not this
  note = job._log_note(log_file)
  if note:
      text += note

  # but this
  if job.log_is_truncated(log_file):
      text += LOG_TRUNCATION_MESSAGE
  ```

- Aim for 20 lines. A reviewer flags a function or method over 40 lines, unless it:

  - makes many calls to other functions,
  - does a lot of printing,
  - renders something, or
  - sits at the edge of the codebase: it is the lowest call in the stack trace, or it is only
    called in one specific scenario.

  Test functions are exempt, because a long test is normal.

- Keep the cyclomatic complexity (the number of paths through a function) at 10 or below.
- Keep high-level and low-level code apart. A high-level function calls other functions to do its
  work. For example, a function in the connection router could be:

  ```python
  recipients = get_recipients()
  encrypted_msg = encrypt(message)
  signed_and_encrypted_msg = sign(encrypted_msg)
  send(signed_and_encrypted_msg)
  ```

  A low-level function does the work itself, such as writing a set of files and their permissions
  to a folder. When a high-level function needs a low-level result, keep that logic minimal or move
  it into a helper. For example, `ab = Y.Z()` followed by logic on `ab` belongs in a helper.

- Give a function or method one responsibility. When it clearly does two things, split it.

### Consistency

- Do not write the same code twice. When two functions do substantially the same work, keep one and
  call it from both places. A repeated line or two is fine: the copy has to be big enough that a
  helper is worth it.
- Keep one source of truth for each piece of state. A cache is the exception.
- Do not use old names, the names of things that are no longer used.
- Do not give a new name to something that already has one.
- A field, name or label means one thing, in data structures and in what users see:
  - Not something else in another case. The "Approval" row of a job says how it was approved:
    manual or auto. An enclave job should not use that row to say "every data owner".
  - Not two things at once. A row that says "every data owner approved; nothing was requested"
    mixes whether the job was approved with which files were requested.

### Drive API calls

When we communicate over Google Drive (and maybe Microsoft 365 later), we want to:

1. make as few API calls as possible,
2. do as much as possible in parallel, and
3. make sure the system recovers from a failed API call or a closed notebook.

This keeps the system fast enough, lets us recover the state when we log back in, and means we never
lose data. Take these three into account whenever you add an API call. In practice:

- List a folder once, then download its files in parallel. Do not list the folder again for each
  file.
- Reuse ids you already have, such as cached folder ids or a file's known parent, instead of
  fetching them again.
- When you change a code path that calls Drive, count its calls before and after the change.

### Error handling

When something goes wrong, be defensive and raise by default. Sometimes there is a reason not to,
for example in a loop that should finish before it raises.

### State

When you add new state, check whether it needs a version, so that the state of a syftbox can be
restored correctly. A cache does not need one: it is fine when a cache starts empty.

### Tests

- Tests cover the use cases we care about, and the amount of tests fits: do not add many tests for
  things that never happen. Security-related code needs good coverage.
- Write both a unit test and an integration test for a change when you can. Unit tests run in
  memory and are fast. Integration tests run against Google Drive and are slow.
- A test name says what the test does, so you know it without reading the test. A name of 5 or 6
  words is fine. Drop articles and padding words: `test_no_hint_when_do_owns_no_jobs`, not
  `test_no_hint_when_the_do_owns_none_of_the_jobs`.
- Put tests in the right file. The tests for one component get at least their own file, and maybe
  their own folder.
- Integration tests call the real Drive API, so keep the number of calls low. One test may check
  several things, as long as they all belong to a single flow.

Look at the existing tests for how to set up a pair of clients. Other connections have their own
methods. Unit tests use `SyftboxManager.pair_with_in_memory_connection`:

```python
ds_manager, do_manager = SyftboxManager.pair_with_in_memory_connection(
    use_in_memory_cache=True,  # or False for filesystem cache
    sync_automatically=False,  # Control sync timing in tests
)
```

Integration tests use `SyftboxManager.pair_with_google_drive_testing_connection`:

```python
ds_manager, do_manager = SyftboxManager.pair_with_google_drive_testing_connection(
    do_email=EMAIL_DO,
    ds_email=EMAIL_DS,
    do_token_path=token_path_do,
    ds_token_path=token_path_ds,
)
```

Integration tests need OAuth tokens in the `credentials/` folder, and the env vars
`AI_AUDIT_EMAIL_DO` and `AI_AUDIT_EMAIL_DS`. `CONTRIBUTING.md` explains how to create the tokens.

## Low level

### Naming

- Use established terms, and the terms the code already uses. When the code says
  `read_permissions` everywhere, do not call it `disclosures` now.
- A function or method name says precisely what it does, so you know what it does without reading
  it. `grant_review_folder_read()` that grants specific files in a folder, not the folder, breaks
  this rule. A name that mentions only one of two things a function does is usually wrong too,
  unless the second thing is very small or part of the first.
- Put the subject in a function or method name: prefer `client.submit_job()` over
  `client.submit()`. From the caller's side, a reader should know what a function does without
  opening it. This matters even more with agentic coding, because we often no longer read every
  line. A user-facing API may sometimes use a shorter name, such as `client.submit()`, although
  `create_dataset()` and `submit_job()` do not. When you change or introduce a user-facing API, call
  it out in the PR description.
- Variable names are precise, but brief. When a folder is the review folder, name it
  `review_folder`, not `folder`.
- Drop articles and padding words from every name. Length is fine when every word earns it.
- Functions and variables use `snake_case`, classes use `PascalCase`, and constants use
  `UPPER_SNAKE_CASE`. Private methods start with an underscore: `_helper()`.

### Code style

- Prefer simple code that is easy to read. Avoid deeply nested dict and list comprehensions with
  complicated control flow.
- When a condition combines more than two checks, give each check a variable that says what it
  checks, then combine the variables:

  ```python
  has_file = path in self.file_hashes
  in_immutable_collection = any(path.is_relative_to(p) for p in immutable_subpaths)
  denied = self.freeze_private_datasets and has_file and in_immutable_collection
  ```

- Put a repeated or magic value in a module-level constant.
- Build strings with f-strings.
- Keep imports at the top of the file where possible, in this order: standard library,
  third-party, local. An import inside a function is fine only to break a circular import.
- Give every public function type hints.
- Write docstrings in Google style. When you change a function, update its docstring.
- When you debug, add a print or a comment only where it is needed.

`ruff format` in pre-commit handles formatting, including the 88-character line length.
