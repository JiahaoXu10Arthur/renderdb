# Design notes

Why this is shaped the way it is, what was rejected, and what to know before
changing it. The README says what the package does; this says why.

## The rule the whole thing follows

> **Read a factorization out of the data. Never invent one.**

Three consequences, and the third is the one that keeps mattering:

- Where the format gives you a key, use it. `<lora:name:0.8>` carries a name
  and a strength; that is a parse, not a guess.
- Where it does not, and you want to merge two things anyway, that is an
  **assertion**. Assertions belong to the caller, not to a reader.
- Where you cannot read something, **say so explicitly**. Never substitute a
  value that looks ordinary.

The third one is load-bearing because **a shape you have never seen looks
exactly like an absence**. That is not a hypothetical; see the fourth
correction below.

## Decision: match node shapes, never node ids

The script this grew from located things by literal node number — node 1043
was the quality prefix, 1130 the handwritten prompt. That survives neither a
graph re-save (ComfyUI renumbers) nor contact with anyone else's workflow.

Matching is on **shape**: the set of input keys a family of nodes uses. Four
readers ship — `native_loader`, `power_loader`, `bundle`, `inline_syntax` —
and `LORA_READERS` is a list you can append to.

**The registry must be consulted before any pre-filter.** An early version had
a "does this look like a LoRA node" check running ahead of the readers, which
meant a reader anyone else registered could never fire. The extension point
was decorative. If you add a filter, put it after.

## Decision: refuse per field, not per row

An index that drops a file is useless. One that stores a wrong value is worse.
Every readable render gets a row — path, mtime and pipeline fingerprint are
cheap and never ambiguous — and only genuinely unreadable fields go `NULL`.

`lora_status` carries a distinction that a bare empty list destroys: "this
render used no LoRAs" and "I could not read this render's LoRAs" are opposite
facts that both look like zero rows. Collapse them and every query counting
LoRA usage is wrong in a direction nobody notices.

## Decision: a strength that could not be read is `partial`, never `ok`

A LoRA whose strength arrives as a wired-in link or a string leaves the column
`NULL`. Reporting `ok` would claim "name and strength recovered" about the one
column this package exists for.

## Decision: `build()` prunes, and prunes only what it owns

`prune=True` is the default. A renamed or deleted file otherwise leaves its row
behind and every count is quietly too high — an index reporting renders that do
not exist is worse than one merely out of date, and it fails in the direction
nobody checks, the same way an empty LoRA list does.

The pruning is scoped: a row survives if its `path` is not under the root being
built. One database can hold several roots, and indexing one of them must not
delete another's rows. That makes the default safe to leave on, which is the
only reason it can be the default.

## The four corrections, and what they cost

Getting `lora_status` right took four passes, every one found by running over
a real corpus rather than by reasoning:

1. A LoRA-manager graph has one node holding the stack and several
   **consumers** wired to it, each carrying its own empty `{"__value__": []}`
   widget. Counting those as unread reported essentially every render
   `partial`.
2. A trigger-word toggle with words switched on carries a *populated*
   `__value__` too — but its entries are `{"text": ...}`. Words, not adapters.
3. A Power Lora Loader placed on the canvas and never filled in carries no
   LoRA, so it is not a LoRA that failed to read.
4. **The one that mattered.** 43 renders applied every one of their LoRAs as
   `<lora:name:0.8>` inside a plain text field — no filename, no `__value__`,
   no `lora_name`. They were reported `no_lora_nodes`: not a warning, not a
   partial, a confident *there are none here*.

The fourth is a silent false clean, the single failure this design exists to
make impossible, and it sat in the author's own corpus the whole time. The
lesson is not "add another reader". It is that the reader list should be
assumed incomplete, which is why `lora_status` exists at all — and why the
numbers in the README describe one corpus rather than claiming coverage.

A fifth way to produce the same false clean does not need an unknown shape at
all: a registered reader that *raises*. Its exception used to be swallowed and
the node treated as one nothing had read, so unless the prefilter below
happened to recognise the class name, a broken reader could return
`no_lora_nodes` — the most confident answer there is, on the strength of a
crash. A reader that raises is now counted unreadable. A reader that returns
nothing still is not: it may have found an empty node, which is correction 3.
A predicate that raises is not either, because it has said nothing about the
node, and treating it as a LoRA would be correction 1.

Notice also that three of the four were **false alarms on shapes that already
worked**. A synthetic test cannot catch those, because you only write a
synthetic case for a shape you already know about. That is what
`tests/fixtures/real_shapes.json` is for.

## Rejected

**A tag table.** The prototype split prompts on commas into 42,373 rows over
1,510 images. The heuristic breaks on natural-language and mixed prompts, and
it duplicates the one part of the prior art that is genuinely well covered.

**Prompt-source segmentation** — splitting a prompt into "quality prefix /
injected trigger words / handwritten". Genuinely useful, and implemented as a
lookup table of four literal node ids from one saved workflow. The API graph
carries no node titles, so there is no general signal to rebuild it from.
Shipping it would have meant shipping a personal constant dressed as a feature.

**Prompt text as a structured column.** It is stored for search and kept out
of the exact-match columns, because the walk cannot see text a LoRA manager
injects at run time or an upsampler generates. Good enough to grep; not good
enough to compare on.

## Before you change anything

**Claims in the README are runnable.** Every console example matches real
output. Change the output format and you must re-run the examples and update
the README. This has been broken before: a debug print's truncated tail
(`repr(t[-160:])` turning `red eyes` into `yes`) once ended up documented as a
literal string the tool had collected.

**Numbers are censuses, not samples.** Every statistic in the README comes
from a full scan. An earlier version sampled 300 files and reported 89/6/4
where the census said 85.6/8.0/4.9 — and the census took seconds. If you can
count everything, count everything.

**Two fixture kinds, and they catch different things.**

- Synthetic graphs prove a reader handles the shape it was written for.
- `tests/fixtures/real_shapes.json` — seven graphs lifted from real renders,
  structure byte-for-byte as ComfyUI wrote it, names scrubbed — proves a
  change did not quietly reclassify the shapes that already worked.

Adding a reader means: write the synthetic case, then run the real fixtures.
If you have a corpus, also rebuild it and check the `lora_status` distribution
has not drifted.

**The chunk layer is tested separately.** The graph fixtures start from parsed
JSON and never exercise the PNG walker. `_realistic_png()` in the test file
mirrors the layout real files have — IHDR, one or two `tEXt` chunks keyed
`prompt` and `workflow`, several IDAT, IEND — because the older synthetic PNGs
had no IDAT to step over and one text chunk to choose from, which is the easy
case.

**Zero third-party dependencies is a hard constraint.** PNG chunk parsing is
hand-written specifically to avoid Pillow. Check with:

```console
python -I -c "
import sys; sys.path.insert(0, '.')
before = set(sys.modules)
import renderdb, renderdb.workflow
new = [m for m in set(sys.modules) - before
       if 'site-packages' in str(getattr(sys.modules[m], '__file__', '') or '')]
print('third-party:', new or 'none')"
```

**One corpus is not the ecosystem.** The corpus these numbers come from
contains **zero** native `LoraLoader` nodes — the most common way to apply a
LoRA in stock ComfyUI. That is exactly why the prototype missed the shape for
so long. A new shape must be covered by a synthetic fixture; a real corpus can
only tell you what it happens to contain.
