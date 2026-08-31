# renderdb

[![test](https://github.com/JiahaoXu10Arthur/renderdb/actions/workflows/test.yml/badge.svg)](https://github.com/JiahaoXu10Arthur/renderdb/actions/workflows/test.yml)

Your ComfyUI output folder is already a complete experiment record. This makes
it queryable.

Every PNG ComfyUI writes carries the API graph that produced it — every seed,
model, LoRA and strength. None of it is searchable, so in practice nobody
looks. "When did I last run this LoRA at 0.4, and what came out" means opening
files one at a time until you give up.

```console
$ renderdb build ~/ComfyUI/output
indexed 1603, skipped 5 -> renders.db
  skipped     5  has no embedded workflow

$ renderdb sql "SELECT name, strength_model, COUNT(*) FROM render_lora
                WHERE strength_model < 0.5 GROUP BY name, strength_model"
```

```console
$ renderdb diff a.png b.png
  seed         777000  ->  990001
  cfg          5.0  ->  6.0

  2 changes
```

No dependencies, stdlib only. 65 tests, CI on Python 3.9, 3.11 and 3.13.

## The one field that is actually missing elsewhere

Be precise about the gap, because it is narrow.

[infinite-image-browsing][iib] (1.3k stars, actively maintained) already parses
LoRAs out of generation metadata and makes the **name** searchable. Its
`update_image_data.py` does `Tag.get_or_create(conn, i["name"], "lora")` — and
that is the whole of it. **The strength is discarded on that line.** Its
`image` table is six columns wide: `id, path, exif, size, date, exif_edited`,
with the entire metadata as one text blob.

[ComfyUI_PromptManager][pm] (162 stars) stores `workflow_data` as a JSON blob.
[SDMeta][sdmeta] (20 stars) caches metadata for display.

So: *"LoRA is queryable"* is taken. **"At what strength"** is not, and neither
is asking it across a whole folder:

```sql
-- which LoRAs have I only ever run at full strength?
SELECT name FROM render_lora GROUP BY name HAVING MIN(strength_model) = 1.0;

-- everything from one pipeline, at one seed, sorted by when
SELECT file, cfg, steps FROM render
WHERE fingerprint = ? AND seed = 777000 ORDER BY mtime;
```

If a browser with a UI is what you want, use one of the above — this has no UI
and never will.

## Node shapes, not node ids, and the bug that proves why

Nothing here matches on a node id. The script this grew from located things by
literal node number — `1043` is the quality prefix, `1130` the handwritten
prompt — which works until ComfyUI renumbers the graph on save, and never
worked on anyone else's workflow at all.

What it matches on is a **shape**: the input keys a family of nodes uses.
Four ship:

| shape | looks like |
|---|---|
| `native_loader` | stock `LoraLoader`: flat `lora_name` + `strength_model` |
| `power_loader` | rgthree Power Lora Loader: `lora_1`, `lora_2`, … dicts |
| `bundle` | LoRA-manager: a whole stack in one widget under `__value__` |
| `inline_syntax` | A1111 text: `<lora:name:0.8>` inside a string field |

The original understood **only** the bundle shape — because that is what its
author's workflow used. Across 1,603 of their renders there is not one native
`LoraLoader`, so nothing ever signalled the gap. Anyone on a stock ComfyUI
install would have got LoRA names with no strengths and no error: the exact
field the tool exists to provide, silently absent, on the most common setup in
the ecosystem.

That is why `LORA_READERS` is a list you can append to:

```python
from renderdb import workflow
workflow.LORA_READERS.insert(0, (my_predicate, my_reader))
```

An unrecognised shape is recorded as unresolved. It is never guessed at.

`tests/fixtures/real_shapes.json` holds seven graphs lifted from real renders —
structure byte-for-byte as ComfyUI wrote it, names scrubbed — covering every
LoRA node class the corpus contains, including an empty Power Lora Loader and
the inline-syntax loader. They exist because a synthetic fixture can show that
a new reader handles a new shape and can never show that it left the old ones
alone, which is how three of the four corrections below became necessary.

## Refuse per field, not per row

An index that drops a file is useless. One that stores a wrong value is worse.
So every readable render gets a row — path, mtime and pipeline fingerprint are
cheap and unambiguous — and only genuinely unreadable fields go `NULL`.

`lora_status` carries the distinction that a bare empty list destroys:

| status | meaning |
|---|---|
| `ok` | every LoRA node was read |
| `partial` | some were read; at least one shape had no reader |
| `unsupported_node_shape` | LoRA nodes are present and none could be read |
| `no_lora_nodes` | there are none — an empty list here is a fact |

Without it, "this render used no LoRAs" and "I could not read this render's
LoRAs" both look like zero rows, and every query counting LoRA usage is quietly
wrong in a direction nobody notices.

Getting that status *right* took four corrections, every one found by running
over 1,603 real files rather than by reasoning about it:

- A LoRA-manager graph has one node holding the stack and several **consumers**
  wired to it — a loader, a trigger-word toggle — each carrying its own empty
  `{"__value__": []}` widget. Counting those as unread reported essentially
  every render as partial.
- A trigger-word toggle with words toggled on carries a *populated*
  `__value__` too, but its entries are `{"text": ...}` — words, not adapters.
- A Power Lora Loader placed on the canvas and never filled in carries no LoRA,
  so it is not a LoRA that failed to read.
- **The one that mattered.** 43 renders applied every one of their LoRAs as
  `<lora:name:0.8>` inside a plain text field — no filename, no `__value__`,
  no `lora_name`. They were reported `no_lora_nodes`: not a warning, not a
  partial, a confident *there are none here*. A silent false clean is the
  single failure this design exists to make impossible, and it was sitting in
  the author's own corpus the whole time.

The corpus now reports **1,447 `ok`, 156 `no_lora_nodes`, zero unreadable** —
15,376 LoRA rows, every one with a strength.

The lesson is in the fourth one, and it is not "add another reader". A shape
you have never seen looks exactly like an absence, so the honest posture is to
assume the reader list is incomplete: `lora_status` exists because it will be,
and the numbers above are what one corpus happens to contain, not a coverage
claim.

## What is deliberately not here

**Prompt text is advisory.** It is stored for search and kept out of the
exact-match columns. The walk cannot see text a LoRA manager injects at run
time or a prompt upsampler generates, so what it returns may be a fraction of
what the model received. Good enough to grep; not good enough to compare on.

**No tag table.** The prototype split prompts on commas into 42,373 rows over
1,510 images. That heuristic breaks on natural-language and mixed prompts, and
it duplicates the one part of the prior art that is genuinely well covered.

**No prompt-source segmentation.** The prototype could split a prompt into
"quality prefix / injected trigger words / handwritten" — genuinely useful, and
implemented as a lookup table of four literal node ids from one saved workflow.
The API graph carries no node titles, so there is no general signal to
reconstruct it from. Shipping it would have meant shipping a personal constant
dressed as a feature.

**A strength that could not be read is `partial`, never `ok`.** A LoRA whose
strength arrives as a wired-in link or a string leaves the column `NULL`.
Calling that `ok` would claim "name and strength recovered" about the one
column this package exists for.

**Model identity is size and mtime, or `None`.** A filename is not an identity:
two people's `anime_v3.safetensors` are different files, and yours changes when
you re-download it. Weak, but checkable — and `None` when the file is not found
is an honest answer where a guess is not.

## Commands

```console
renderdb build <dir> [--db renders.db]     index a folder (prunes rows for
                                           files no longer there)
renderdb why <image.png> [--models <dir>]  how one render came to be
renderdb diff <a.png> <b.png>              what changed between two
renderdb stats [--db renders.db]           overview, LoRA usage, strength spans
renderdb sql "SELECT ..." [--db ...]       anything else
```

`diff` reports whether exactly one thing changed, and refuses to call a
comparison single-variable when either render's LoRA stack could not be read —
silence about what was unreadable is how a comparison gets trusted further than
it should be.

## Install

Not on PyPI. From a clone:

```console
pip install .          # or -e ".[test]" to run the suite
pytest -q
```

See [DESIGN.md](DESIGN.md) for why it is shaped this way, what was
rejected, and what to check before changing it.

## License

MIT

[iib]: https://github.com/zanllp/sd-webui-infinite-image-browsing
[pm]: https://github.com/ComfyAssets/ComfyUI_PromptManager
[sdmeta]: https://github.com/jamesmoore/SDMeta
