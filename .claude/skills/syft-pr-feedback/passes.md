# The three passes

Each pass reads `/tmp/pr-<N>/` (the diff, `body.md`, `context/review-rules.md`,
`context/invariants.md`, `context/existing-comments.md`) and the summary from step 2. It reads code
only, changes nothing, and returns candidates. The main run verifies them, so a pass may propose a
candidate it is not sure of, but it says what it checked.

Judge only lines the pull request added or changed. Read the code around them only to understand
them.

## Candidate format

Return a list. Each candidate has these six lines, and nothing else:

```
- rule: <rule id, such as D3, an invariant id, such as SYNC-2, a quickfix tag, or BUG>
  place: <path>:<line> · <Class.method() or file.py: function()>
  does: <what the code does, one sentence>
  wrong: <what is wrong, with the concrete case: the caller, the input, the order of events>
  fix: <the change, concrete enough to apply without asking>
  checked: <what you read to confirm it: callers, readers, a test, the ticket>
```

Use `BUG` for a concrete defect that no rule names. Return `none` when nothing is wrong.

## Pass A — Design

Rules: D1, D2, D3, D4, D7, D8, U1–U4, C1, C2, S1, S4, and every invariant in
`context/invariants.md`.

Ask of every new or changed class, function and flow:

- Do we need it? What breaks if we delete it? Which caller takes this branch?
- Does it work around a problem that we could fix instead?
- Is this fact already stored somewhere else? Can it be looked up instead of copied?
- Which layer owns this? Does a manager or a client decide something a storage layer or codec should?
- In what order does the flow do its steps? How many Drive calls does it make per peer or per file?
- Is the data a typed model with a shape that allows the general case?
- What does the data owner see, and what does the data scientist see? Is the default safe, and can
  the user override it from the API they call?
- Is this backwards compatibility that the current phase does not want (C1)?
- Does it match the ticket or design the body links?

## Pass B — Correctness and safety

Rules: E1–E5, D5, D6, every invariant in `context/invariants.md`, and BUG.

For every changed path, try to break it, and write down the steps when you can:

- Which input, empty or missing file, order of events, second process or failed network call makes
  it wrong?
- Is an exception swallowed, or caught so widely that a bug looks like a normal case? On a security
  path, what does the fallback trust?
- Is a file that holds keys, settings or state written atomically?
- Does a value from Drive or from a peer's file decide something without a check?
- Does a new kind of message, protocol or state fall into an `else` that handles it silently?
- Does the code reach into a private attribute of another object?
- Is an interface bigger than it needs to be, or a class only forwarding calls?

When the fix is short, put the fixing code in `fix`.

## Pass C — Readability: names, tests and docs

Rules: N1–N5, T1–T5, W1–W4, S2, S3, and the ten checks under **Checks** in
`.claude/skills/syft-pr-quickfix/SKILL.md`, with their tags.

- Names: does each new name say what it acts on, match what the code does and how, and use the
  term the codebase or the field already uses? Respect the exceptions in N1.
- Tests: does a regression test fail on the old code? Is it fast? Is one test redundant with
  another? Does a test check behaviour that should not exist? Is the risky case tested?
- Docs and comments: is every term introduced before it is used? Does the example still run? Is
  hard logic explained, and is every comment still true?
- Hygiene: is anything temporary left in, such as commented-out code or debug output? Is an import
  inside a function, or a function nothing calls?
