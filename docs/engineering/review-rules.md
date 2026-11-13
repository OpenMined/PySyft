# Review rules

The rules a pull request in this repository is reviewed against. A person applies them, and so
does the `syft-pr-feedback` skill. The facts about the system that a change must keep true are in
`.claude/rules/invariants/`; this file is about how we write the code.

## How to read a rule

Every rule has an id, such as `D3`. A finding names the id, so a rule can be discussed, changed or
dropped.

- **Status:** `candidate` was mined from past reviews and is not yet confirmed. `active` is
  confirmed by the team. `dropped` stays here for the record and is not checked.
- **Source:** K is Koen, P is Pedro. "Seen" quotes the review comments the rule came from.
- **Severity:** what a finding against the rule is, unless the finding shows more harm.
- **Flag** says what to report. **Not** says what looks similar and is fine.

## Severity

- 🔴 **Must fix** — breaks an invariant, or a concrete input, order of events or failure makes the
  code wrong, insecure or lose data.
- 🟡 **Should fix** — breaks a rule here, and the fix is concrete.
- ⚪ **Nit** — a name, a comment or doc wording. At most 5 in one review.
- ➜ **Follow-up** — a real problem outside the scope of the pull request. It becomes a ticket.

## Do not report

- Code the pull request did not add or change.
- What ruff, pre-commit or CI already catches, such as formatting and import order.
- A repeated single line. A duplicate is worth a finding only when it is more than one line.
- The length of a test.
- "Consider X" without a concrete fix.
- A point that a comment on the pull request already makes.

## Changing a rule

Change this file in a pull request. The skill reads the rules from the base branch, so a pull
request cannot change the rules of its own review. Rules come from the feedback recorded in
`.claude/skills/syft-pr-feedback/feedback.md`; a rule becomes `active` only when the team agrees.

---

## D — Design and complexity

### D1 · Complexity has to earn its place

`candidate` · K · 🟡

- **Flag:** code for a scenario that does not happen or that we do not care about; a second code
  path for one job; machinery such as a queue, retries, a cache or a pre-check whose need the pull
  request does not show.
- **Not:** a branch that a caller in the repository really takes.
- **Seen:** PR 9562 "Do we need this complexity?", "This again does a lot of handling of scenarios
  we don't care about", "Why do we have 2 code paths here, do we need that?", "Maybe don't queue,
  but warn and forget?" · PR 9497 "Does this pre-check really solve anything?" · PR 9436 "we are
  not using these, remove".
- **Fix:** delete it, or name the caller that needs it.

### D2 · Fix the cause, not the symptom

`candidate` · K · 🟡

- **Flag:** code that works around a problem, such as a hint, a pre-check or state copied on every
  call, where the problem itself can be fixed.
- **Seen:** PR 9497 "fixing the hints kind of solve a problem that shouldnt exist in the first
  place" · PR 9562 "This feels like a code smell. Are we doing something wrong here?"
- **Fix:** name the cause and the change that removes it.

### D3 · One source of truth

`candidate` · K, P · 🟡

- **Flag:** a fact stored in two places that can drift apart: a field, a file or a parameter that
  repeats what another one holds.
- **Not:** a cache that is rebuilt from its source.
- **Seen:** PR 9604 "disclosures are stored in a file, that could be implied from the permissions so
  it's duplicate information that we need to keep in sync" · PR 9487 "we dont need multiple sources
  of truth" · PR 9438 (P) "Why are we taking `folder` and `email` as parameters here, instead of
  using attributes from sync? It looks like a potential source of drift."
- **Fix:** keep one place, and look the fact up from it.

### D4 · Each layer does its own job

`candidate` · K, P · 🟡

- **Flag:** logic in the wrong layer: a manager or client that decides a disk layout or branches on
  a protocol version; dataset logic in syft; work that belongs to the storage layer done by its
  caller.
- **Seen:** PR 9443 (K) "it feels like this should be part of the codec", "the responsibility of
  this should probably lie with the storage layer" · PR 9443 (P) "The manager should never make
  internal decisions based on protocol version."
- **Fix:** move it into the layer that owns it, and name that layer.

### D5 · Do not reach into another object's private attributes

`candidate` · P · 🟡

- **Flag:** `a._b` or `a._b._c` used from outside the class of `a`.
- **Seen:** PR 9438 "The client is reaching for private attributes of its children ... If that's
  intended, there's no point in prefixing these attributes with `_`."
- **Fix:** a public method or property on the owner.

### D6 · Small contracts

`candidate` · P · 🟡

- **Flag:** an abstract method that other abstract methods can build; a class that only forwards
  calls to another one.
- **Seen:** PR 9561 "`fetch_all_events()` is `fetch_events_since(None)`", "`OwnerStorage` and
  `WatcherStorage` are wrappers that only forward calls to `PrivateStore`."
- **Fix:** keep the primitives abstract, and make the rest concrete helpers on the base class.

### D7 · A flow does its steps in a clear order, and batches its I/O

`candidate` · K · 🟡

- **Flag:** steps that interleave or repeat, such as sending, pulling and sending again; one Drive
  call per item where one call can do many; per-peer work done one peer at a time where it can run
  in parallel; a whole file rewritten for every small change.
- **Seen:** PR 9571 "Why are we first delivering things, then pulling changes and delivering again,
  is that necessary? This doesn't help for reducing the number of API calls" · PR 9562 "it seems
  like this is quite slow, because we do it for every peer and it's not in parallel" · PR 9571
  "should be a sqlite and not a json".
- **Fix:** the order the flow should have, or the batch call to use.

### D8 · Data has a type and the right shape

`candidate` · K · 🟡

- **Flag:** a `dict` read or returned where a model fits; a signature that cannot express the
  general case.
- **Seen:** PR 9562 "why does `_read_key_file` return a dict? That should be typed" · PR 9604
  "`release()` ... should ideally be a dict with filenames as keys and a list of recipients as
  values."
- **Fix:** the model or the type to use.

## E — Errors

### E1 · Raise instead of returning False or None on failure

`candidate` · K · 🟡 (quickfix tag `silent-fail`)

- **Flag:** a function that returns `False` or `None` to say it failed, so callers fail later or
  not at all.
- **Seen:** PR 9562 "Can we not just raise?"
- **Fix:** the exception to raise, and where.

### E2 · Catch specific exceptions

`candidate` · P · 🟡, 🔴 on a security path

- **Flag:** `except Exception` or a bare `except` that hides a bug, especially where the fallback
  trusts something: a missing key, a default policy, a first use.
- **Not:** a top-level loop that logs the error with its context, as in E3.
- **Seen:** PR 9520 "This bare `Exception` is bad, not just as a matter of style. A bug should fail
  loudly here for security reasons." · PR 9438 "If a network request fails, it's going to behave as
  if it's already shared." · PR 9443 "This will silently ignore errors due to unreadable datasets."
- **Fix:** the exceptions to catch, such as `HttpError` or `ValueError`. A legitimate ignore uses
  `contextlib.suppress(...)` with a specific exception, and logs.

### E3 · One bad item does not block the others

`candidate` · K · 🟡

- **Flag:** a loop over peers, messages or proposals where one failure stops the rest, or retries
  without a limit.
- **Seen:** PR 9571 "I wonder whether this is a good idea if one broken message blocks the rest."
- **Fix:** log the error with the item, go on, and retry a bounded number of times.

### E4 · An unknown case fails loudly

`candidate` · P · 🟡

- **Flag:** an `else` that silently handles every kind not named before it.
- **Seen:** PR 9564 "This `else` clause feels fragile ... so it doesn't fail silently in case a new
  kind is added in the future."
- **Fix:** an explicit branch per known kind, and a raise for the rest.

### E5 · State and key files are written atomically

`candidate` · P · 🔴

- **Flag:** `write_text()` or `open(..., "w")` on a file whose loss or corruption locks a user out
  or resets trust, such as keys, settings or persisted state.
- **Seen:** PR 9520 "`write_text()` is not safe for this. If the write crashes ... the private keys
  can't be read back."
- **Fix:** hold a lock over read, change and write; write a temporary file, `fsync`, then
  `os.replace()`, as `syft/sync/sync/caches/persisted_dict.py` does.

## U — Users

### U1 · Meet users where they are

`candidate` · K · 🟡

- **Flag:** a user-facing behaviour that Python users do not expect, such as `sys.exit()` instead of
  an exception, or a handle that depends on position instead of a stable key.
- **Seen:** PR 9577 "we should meet the user where they are (what they are used to), which is
  raising exceptions" · PR 9497 "we need a deterministic way to get job X from user Y, ideally
  `client.jobs["a@b.org"]["<jobname>"]`."
- **Fix:** the behaviour the user expects.

### U2 · A safe default, with an override in the top-level API

`candidate` · K · 🟡

- **Flag:** a risky default, such as accepting an incompatible peer or writing every protocol, or a
  safe default that the user cannot override from the API they call.
- **Seen:** PR 9487 "I think we should refuse by default, but it should be possible for the user to
  override this. I think in general this should often be the case" · PR 9443 "we should probably by
  default always write in the current version", "perhaps we should also have the posibility to pass
  in protocol versions here in case we want to override something."
- **Fix:** the default, and the parameter that overrides it.

### U3 · Check what the data owner and the data scientist each see

`candidate` · K · 🟡

- **Flag:** a folder, a state or a message that confuses one of the two roles.
- **Seen:** PR 9604 "Isn't this confusing from a DO perspective? Let's say I ran a job, now I want
  to read files, but now the folder I am expecting is empty."
- **Fix:** what that role should see.

### U4 · When a step is needed, do it and say so

`candidate` · K · 🟡

- **Flag:** a message that tells the user to run a step that the code could run itself.
- **Seen:** PR 9487 "If the answer is 'keep everything and repair on the next sync', we should
  probably just call `sync()` right there, and print that we are doing it."
- **Fix:** run the step and print one line about it.

## N — Names

`syft-pr-quickfix` checks names too (`vague-name`, `name-mismatch`, `old-concept`). These rules add
what it does not cover.

### N1 · A name says what it acts on

`candidate` · K · ⚪

- **Flag:** a bare verb or a generic word: `submit()`, `deliver()`, `confirm()`, `ensure_keys()`,
  `state`.
- **Not:** a user-facing API where a short name reads well, such as `client.create`; a simple
  container method, such as `get` on a mapping.
- **Seen:** team meeting "I would usually go for submit job" · PR 9571 `deliver` →
  `deliver_file_changes`, `confirm` → `confirm_deliveries` · PR 9562 "`ensure_keys` is a bit vague
  as a name. We should probably mention 'own keys'."
- **Fix:** the name with its object.

### N2 · A name matches what the code does, and how

`candidate` · K · ⚪

- **Flag:** a name that implies another mechanism or another effect.
- **Seen:** PR 9487 "`negotiate_protocol_version` sounds like we are actively communicating with the
  peer, which is not what's happening. Something like `infer_protocol_version_to_use` makes more
  sense" · PR 9604 "`grant_ds_review_read` is a weird function name, because it describes something
  it doesn't do."
- **Fix:** the name that says what happens.

### N3 · One concept, one name, and the established one

`candidate` · K, P · ⚪

- **Flag:** an invented term where a common one exists; a new name for a concept the codebase
  already names; an existing name reused for something else; one word with several meanings in one
  file.
- **Seen:** team meeting "if it's a queue, use a queue" (not "spool") · PR 9604 "Disclosures are just
  another name for read permissions, so let's be consistent and use that", "release may also mean
  'make public' and 'share' implies 'with someone'" · PR 9561 (P) "The name `PeerState` is already
  used by the enum" · PR 9438 (P) "The word 'sync' is carrying four different meanings in the file."
- **Fix:** the established or existing name.

### N4 · A boolean says what it means

`candidate` · K · ⚪

- **Seen:** PR 9562 "`use_encryption` is renamed to `requires_encryption`, which says what it
  means."
- **Fix:** the name that reads as a yes or no question about one fact.

### N5 · A test name carries no filler

`candidate` · K · ⚪

- **Seen:** PR 9497 `test_no_hint_when_the_do_owns_none_of_the_jobs()` →
  `test_no_hint_when_do_owns_no_jobs()`.
- **Fix:** the name without articles and filler words.

## C — Compatibility and releases

### C1 · No backwards compatibility in this phase

`candidate` · K · 🟡 · Applies from the migration reset (November 2026) until the team decides
otherwise.

- **Flag:** read paths, seeding, upgrade code or "outdated peer" handling for files or peers from
  before the current release, and the tests for them.
- **Seen:** PR 9562 "I think we dont need to be backwards compatible for the upcoming release",
  "No need for backwards compatibility, we can just assume the policy is there" · PR 9571 "We don't
  need backwards compatibility for existing clients, so we can remove both the seeding ... and the
  reading of `owner_pending_deliveries.json`."
- **Fix:** delete the path and its tests.

### C2 · Release files are left to the release job

`candidate` · K · 🟡

- **Flag:** a version changed by hand; a release fixture or artifact for a version that is not
  released; compatibility that is kept but not tested with release fixtures.
- **Seen:** PR 9443 "is this necessary?" (a version bump), "this whole folder should probably not
  exist because this version has not been released yet" · PR 9487 "Until we have migrations for them
  we should probably have a test that we can read old versions?"
- **Fix:** revert the file, or add the fixture test.

## T — Tests

### T1 · A regression test fails without the fix, and runs fast

`candidate` · K · 🟡

- **Flag:** a test for a bug fix that also passes on the old code, or that goes through a slow flow
  where a direct call works.
- **Seen:** PR 9529 "It also failed the same way without the fix, so it didn't catch the bug", "It
  was slow (~7s), because it went through the full submit → approve → venv flow."
- **Fix:** the direct call to test, and the assertion that fails on the old code.

### T2 · No redundant tests

`candidate` · K · ⚪

- **Flag:** a test that another test already covers; one test per version where one loop over all
  versions works.
- **Seen:** PR 9443 "this should be more generic, it should just load every package artifact that
  was released", "we can remove this, we already have the one that loads all of them."
- **Fix:** the test to keep, and the ones to delete or merge.

### T3 · A test of removed behaviour goes with it

`candidate` · K · 🟡

- **Flag:** a test that checks behaviour that should not exist, such as an enclave without
  encryption or a format from before the current release.
- **Seen:** PR 9562 "The enclave should never run without encryption, so a test that sets the
  enclave to `off` can be deleted."
- **Fix:** delete it.

### T4 · Test the risky case

`candidate` · K, P · 🟡

- **Flag:** a change to a security or migration path without a test of the case that can go wrong:
  encryption on, the attack itself, a downgrade.
- **Seen:** PR 9487 "we probably need a migration test with encryption on as well" · PR 9520 (P)
  "pin Bob, sync with Bob rejected, sync with Bob accepted and a new key, then assert that the old pin
  stays."
- **Fix:** the test, in one sentence: setup, action, assertion.

### T5 · Related tests sit together

`candidate` · K · ⚪

- **Seen:** PR 9497 "Perhaps we should isolate the hint tests in their own file."
- **Fix:** the file to move them to.

## W — Docs and comments

### W1 · Introduce a term before using it

`candidate` · K · ⚪

- **Seen:** PR 9436 "concept of binding table not introduced yet, describe without it", "checker,
  not introduced yet."
- **Fix:** a plain description, or the definition first.

### W2 · Plain words, and no "you" in reference docs

`candidate` · K · ⚪

- **Seen:** PR 9436 "I dont thing using 'You' is the tone of voice we want", "maybe its me, but I am
  not familiar with the term 'knob'."
- **Fix:** the plain or established word.

### W3 · A doc matches the code, and shows one way

`candidate` · K · ⚪

- **Flag:** an example that the code no longer runs; two documented ways to do one thing.
- **Seen:** PR 9436 "can you make sure this aligns with the example, I think currently this is out
  of sync", "having two UX's described here complicates the story."
- **Fix:** the line to change, or the way to drop.

### W4 · Hard logic is explained, and comments stay true

`candidate` · K, P · ⚪

- **Flag:** logic that takes several readings to follow and has no comment; a comment that
  contradicts the code.
- **Seen:** PR 9562 "This logic is hard to follow, add comments" · PR 9436 "some comments here
  would help me or otherwise split it up in functions" · PR 9438 (P) "This comment seems
  contradictory."
- **Fix:** the comment to add, or the function to split out.

## S — Scope and hygiene

### S1 · Out of scope is a follow-up

`candidate` · K · ➜

- **Flag:** a real problem that the pull request does not need to fix.
- **Seen:** many threads end with "out of scope for this PR" and a ticket.
- **Fix:** a ticket title and one line of body.

### S2 · Nothing temporary is merged

`candidate` · K · 🟡

- **Flag:** commented-out code, debug output, a skipped test without a reason, a TODO without a
  ticket.
- **Seen:** PR 9339 "can we remove before merge?"
- **Fix:** delete it, or link the ticket.

### S3 · Imports at the top, and no dead code

`candidate` · P · ⚪

- **Flag:** an import inside a function (fine only to break an import cycle, with a comment); an
  import repeated; a function, branch or parameter that nothing uses.
- **Seen:** PR 9438 "Redundant import. It's already imported at the top of the file" · PR 9443
  "`_validate_dataset_name` is dead code and should be removed" · PR 9564 "`download_events_message_by_id`
  is not being called by anyone."
- **Fix:** move or delete it.

### S4 · The code matches its ticket

`candidate` · P · 🟡

- **Flag:** a difference between the code and the ticket or design the pull request links.
- **Seen:** PR 9561 "The v1 ticket has the peer registry methods in `PeeringConnection`. It's fine to
  be in `PrivateStore`, but the ticket should be updated to avoid confusion."
- **Fix:** change the code, or say which ticket to update.
