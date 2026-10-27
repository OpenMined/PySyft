# Feedback archive

Every piece of feedback on the `syft-pr-quickfix` skill. Only append to this file. Nothing is
deleted, even after a rule covers it.

`syft-pr-quickfix-iterate` reads this whole file before it writes a rule, so that it notices when
the same complaint comes up a third time. `syft-pr-quickfix` does not read this file.

## Format

    ## YYYY-MM-DD — PR <N>, short label
    Kind: missed | wrong | bad fix | report | new check
    Feedback:
    > what the user said, word for word
    Rule: the one-sentence rule taken from it
    Landed: where it went in syft-pr-quickfix/SKILL.md, or "covered by <existing rule>"

---

## 2026-10-26 — first version

Kind: new check (eight)

Feedback:

> - There are code paths that support multiple scenarios, but those scenarios never happen or are
>   not important to support but complicating the code
> - We are introducing multiple sources of truth, multiple variables that actually represent the
>   same state and could go out of sync. This is not applicable for caches that are properly
>   managed.
> - We are introducing names that are referencing and old concept with a new name
> - Code is unnecessary complicated or long
> - When we create an object, ideally it should be simple and the code should get all the
>   information and then make the object, it shouldn't create a half baked object and then modify
>   it in place a bunch of times
> - If we are doing the same thing twice in multiple places (distributed responsibility)
> - Returning False / None instead of raising
> - Name inconistencies. For instance calling a method upload_versioninfo, while it actually takes
>   a peerinfo object

Rule: one check per item, tagged `unused-case`, `same-state`, `old-concept`, `too-complex`,
`half-built`, `done-twice`, `silent-fail`, `name-mismatch`.

Landed: `## Checks`, 1 to 8.

## 2026-10-27 — PR 9571, only easy fixes

Kind: report (and drops a check)

Feedback:

> oke, I would like to remove 4 from the skill, I only want to keep 5 if its a very easy fix. I
> also want to keep 6 if its a very easy fix. In general I would like the result of this skill to
> be a list of easy fixes, so easy that I dont have to review the suggestions too much. everything
> that is more complicated should go to an appendix when presenting the result

Rule: the main list holds only fixes the user can accept without reviewing them closely; every
other finding goes to an appendix, and `too-complex` is no longer a check.

Landed: `## Checks` (removed `too-complex`, easy-fix condition on `half-built` and `done-twice`),
new `## Easy fixes` section, `## Report` (appendix and example).

## 2026-10-27 — PR 9571, examples list, one-line duplicates, vague names, unnamed values

Kind: wrong (1, 2), report (3), new check (vague names, unnamed values), plus an examples list

Feedback:

> Oke, I want to iterate more on the skill. I want you to keep a local list of good and bad
> examples so the skill can check when in doubt. 1 is not a good example because its just a single
> lines. 3 is fine but not super high priority, 2 is also not good because its also a single line.
> I want to add another rule, while is vague names, I dislike \_held_before_seed because it does not
> explain that its about files. There is also something in \_held_before_seed I dont like, which is
> this line files.update(dict.fromkeys(unsent, UNSENT_HASH)), because I dont know what that
> intermediate variable is

Rules:

- `done-twice` needs a duplicated job of more than one line; a repeated one-liner is not a finding.
- Fixes that only change a docstring or comment go after fixes that change code.
- New check `vague-name`: a name does not say what kind of thing it holds, takes or returns.
- New check `unnamed-value`: a value is built inline inside another call, so the reader cannot
  tell what it is; give it a named variable.
- The skill keeps `examples.md` with good and bad findings and reads it when unsure; the iterate
  skill adds an example for every finding the user judges.

Landed: `## Checks` (`done-twice` text, checks 8 and 9), `## Steps` (step 4 reads `examples.md`),
`## Report` (docstring fixes last), new `syft-pr-quickfix/examples.md`, and step 1 of
`syft-pr-quickfix-iterate/SKILL.md`.

## 2026-10-27 — PR 9571, clickable places

Kind: report

Feedback:

> can you make all the bullits somehow clickable so I easily open the file?

Rule: the place of every finding is a Markdown link to the absolute file path at its line.

Landed: `## Report` (place rule and example report); the numbering rule was shortened to one line
to stay inside the budget.

## 2026-10-27 — PR 9571, links open Finder

Kind: report

Feedback:

> hmm, the link opens finder, not cursor (which is default for these files)

Rule: the place link uses `cursor://file<absolute path>:<line>`, so it opens Cursor at the line.

Landed: `## Report` (place rule and example report).

## 2026-10-27 — PR 9571, bare verb `deliver`

Kind: missed

Feedback:

> Hmmm, so the skill didnt catch the "deliver" function, why? Wouldnt deliver_missing_files be
> better?

Rule: `vague-name` also covers a bare verb that does not say what it acts on.

Landed: `## Checks`, `vague-name` text (same line count), and a good example in `examples.md`.

## 2026-10-27 — PR 9571, generic word `state`

Kind: missed

Feedback:

> It also did catch \_create_current_state_event, I would prefer \_create_current_filestate_event.
> There is so much state that I dont know from the name. update skill and rename in code

Rule: `vague-name` also covers a generic word (`state`, `data`, `info`) that does not say whose.

Landed: `## Checks`, `vague-name` text (same line count; the `_held_before_seed` example moved to
`examples.md` only), and a good example in `examples.md`.

## 2026-10-27 — PR 9571, bare verb `confirm`

Kind: missed

Feedback:

> also, confirm(), shouldnt that be confirm delivery?

Rule: none new.

Landed: covered by the bare-verb part of `vague-name`, added earlier today. Added a good example
to `examples.md`.

## 2026-10-27 — PR 9571, nested dict types

Kind: new check

Feedback:

> Ah, another rule. If we have a data structure like this dict[str, dict[str, str]] as in
> \_files_held_by_peer_at_seed. It should add in the docstrings what it is and in the code a single
> line comment indicating what points to what (peer email to file hash to something else) ro
> whatever. Update skill and write

Rule: a nested type says what its keys and values are, in the docstring and in a one-line
`# key -> {key: value}` comment where it is built or declared.

Landed: new check 10, `unlabelled-mapping`. To stay at 119 lines, `unused-case`, `half-built` and
`unnamed-value` were each shortened by one line.
