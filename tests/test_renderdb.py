import contextlib
import json
import struct
import zlib

import pytest

from renderdb import workflow

from renderdb import (build, compare_renders, connect, model_identity,
                      provenance, scan_one)
from renderdb.workflow import (NONE, OK, PARTIAL, UNSUPPORTED, WorkflowError,
                               fingerprint, loras, models, prompt_text,
                               read_workflow, sampler_settings)


def _png(chunks):
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    for ctype, payload in [(b"IHDR", ihdr)] + list(chunks) + [(b"IEND", b"")]:
        out += struct.pack(">I", len(payload)) + ctype + payload
        out += struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
    return bytes(out)


def _write(path, api):
    path.write_bytes(_png([(b"tEXt",
                            b"prompt\x00" + json.dumps(api).encode())]))
    return path


def wf(prompt="1girl, solo", seed=7, cfg=5.0, ckpt="anime_v3.safetensors",
       lora_nodes=None):
    api = {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "worst quality"}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": 30, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0, "positive": ["6", 0],
                         "negative": ["7", 0]}},
    }
    api.update(lora_nodes or {})
    return api


NATIVE = {"10": {"class_type": "LoraLoader",
                 "inputs": {"lora_name": "style_a.safetensors",
                            "strength_model": 0.8, "strength_clip": 0.6,
                            "model": ["4", 0], "clip": ["4", 1]}}}

BUNDLE = {"20": {"class_type": "Lora Stacker (LoraManager)",
                 "inputs": {"loras": {"__value__": [
                     {"name": "style_b", "strength": 1.0, "active": True},
                     {"name": "style_c", "strength": 0.3, "active": False}]}}}}


# ---------------------------------------------------- the shape that was missed

def test_native_loraloader_yields_name_and_strength():
    """The prototype only understood the bundle shape, because that is what
    its author's workflow used -- across 1,603 of their renders there is not
    one native LoraLoader, so nothing ever signalled the gap. A stock ComfyUI
    setup would have got names with no strengths and no error."""
    got, status = loras(wf(lora_nodes=NATIVE))
    assert status == OK
    assert got[0]["name"] == "style_a.safetensors"
    assert got[0]["strength_model"] == 0.8
    assert got[0]["strength_clip"] == 0.6


def test_bundle_shape_still_works():
    got, status = loras(wf(lora_nodes=BUNDLE))
    assert status == OK
    assert {g["name"] for g in got} == {"style_b", "style_c"}
    assert [g["strength_model"] for g in got] == [1.0, 0.3]


def test_the_active_flag_survives():
    got, _ = loras(wf(lora_nodes=BUNDLE))
    assert [g["active"] for g in got] == [True, False]


def test_both_shapes_in_one_graph():
    merged = dict(NATIVE)
    merged.update(BUNDLE)
    got, status = loras(wf(lora_nodes=merged))
    assert status == OK
    assert len(got) == 3
    assert {g["shape"] for g in got} == {"native_loader", "bundle"}


# ------------------------------------------------------- status is the honest bit

def test_no_lora_nodes_is_different_from_unreadable():
    assert loras(wf())[1] == NONE


def test_an_unreadable_shape_is_not_reported_as_no_loras():
    """An empty list with UNSUPPORTED means 'there are LoRA nodes here and I
    could not read them'. Storing that as 'no rows' would make every query
    counting LoRA usage quietly wrong."""
    odd = {"30": {"class_type": "SomeVendorLoraThing",
                  "inputs": {"lora_config": "vendor_style.safetensors",
                             "weight_spec": "0.5:0.5"}}}
    got, status = loras(wf(lora_nodes=odd))
    assert got == []
    assert status == UNSUPPORTED
    assert status != NONE


def test_partial_when_one_shape_reads_and_another_does_not():
    mixed = dict(NATIVE)
    mixed["31"] = {"class_type": "WeirdLoraStack",
                   "inputs": {"lora_blob": "lora_things.safetensors"}}
    got, status = loras(wf(lora_nodes=mixed))
    assert len(got) == 1
    assert status == PARTIAL


def test_a_new_shape_can_be_registered():
    """The registry is the extension point; the alternative is a package that
    only works on its author's workflows."""
    from renderdb import workflow as W
    marker = {"40": {"class_type": "MyLoader",
                     "inputs": {"my_lora": "x.safetensors", "my_weight": 0.5}}}

    def mine(node):
        return isinstance((node.get("inputs") or {}).get("my_lora"), str)

    def read(nid, node):
        ins = node["inputs"]
        return [{"name": ins["my_lora"], "strength_model": ins["my_weight"],
                 "strength_clip": None, "active": True, "shape": "mine",
                 "node": nid}]

    W.LORA_READERS.insert(0, (mine, read))
    try:
        got, status = loras(wf(lora_nodes=marker))
        assert status == OK and got[0]["strength_model"] == 0.5
    finally:
        W.LORA_READERS.pop(0)


# ------------------------------------------------------------------ structure

def test_sampler_settings_are_read():
    s = sampler_settings(wf(seed=1234, cfg=6.5))
    assert s["seed"] == 1234 and s["cfg"] == 6.5 and s["steps"] == 30


def test_a_seed_wired_in_from_a_node_is_followed():
    """Seeds are usually not literals. Reading only literals would silently
    miss the most important controlled variable."""
    api = wf()
    api["3"]["inputs"]["seed"] = ["99", 0]
    api["99"] = {"class_type": "Seed (rgthree)", "inputs": {"seed": 4242}}
    assert sampler_settings(api)["seed"] == 4242


def test_an_unreadable_seed_is_none_not_a_guess():
    api = wf()
    api["3"]["inputs"]["seed"] = ["98", 0]
    api["98"] = {"class_type": "Mystery", "inputs": {"a": 1, "b": 2}}
    assert sampler_settings(api)["seed"] is None


def test_models_are_read():
    assert models(wf(ckpt="x.safetensors"))["ckpt_name"] == "x.safetensors"


def test_fingerprint_ignores_widget_values_but_not_wiring():
    """Two renders with the same fingerprint went through the same machine.
    That is what makes 'did I change the pipeline or the prompt' answerable."""
    assert fingerprint(wf(prompt="a", seed=1)) == \
        fingerprint(wf(prompt="b", seed=2))
    assert fingerprint(wf()) != fingerprint(wf(lora_nodes=NATIVE))


def test_prompt_text_is_positive_only():
    assert prompt_text(wf(prompt="1girl, sunset")) == "1girl, sunset"
    assert prompt_text(wf(), "negative") == "worst quality"


def test_prompt_text_gives_up_quietly_because_it_is_advisory():
    """Unlike the structured columns, this one may be a fraction of what the
    model got, so it is search-only and never compared on."""
    api = wf()
    api["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": ["77", 0]}}
    api["77"] = {"class_type": "TriggerWord Toggle (LoraManager)",
                 "inputs": {"x": 1}}
    assert prompt_text(api) is None


# ---------------------------------------------------------------- reading PNGs

def test_scan_one(tmp_path):
    p = _write(tmp_path / "a.png", wf(lora_nodes=NATIVE))
    row = scan_one(p)
    assert row["checkpoint"] == "anime_v3.safetensors"
    assert row["seed"] == 7 and row["cfg"] == 5.0
    assert row["lora_status"] == OK
    assert row["loras"][0]["strength_model"] == 0.8


def test_a_png_without_a_workflow_raises(tmp_path):
    p = tmp_path / "plain.png"
    p.write_bytes(_png([]))
    with pytest.raises(WorkflowError):
        scan_one(p)


def test_not_a_png(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"nope")
    with pytest.raises(WorkflowError):
        read_workflow(p)


# ------------------------------------------------------------------- indexing

def test_build_indexes_and_skips(tmp_path):
    _write(tmp_path / "a.png", wf(seed=1, lora_nodes=NATIVE))
    _write(tmp_path / "b.png", wf(seed=2, lora_nodes=BUNDLE))
    (tmp_path / "bad.png").write_bytes(_png([]))
    n, skipped = build(tmp_path, tmp_path / "db.sqlite")
    assert (n, skipped) == (2, 1)


def test_lora_strength_is_queryable_as_a_column(tmp_path):
    """The whole point: a 1,341-star browser already makes the LoRA *name*
    searchable as a tag and throws the strength away at the line that reads
    it."""
    _write(tmp_path / "a.png", wf(lora_nodes=NATIVE))
    db = tmp_path / "db.sqlite"
    build(tmp_path, db)
    c = connect(db)
    row = c.execute("SELECT name, strength_model FROM render_lora").fetchone()
    assert row["name"] == "style_a.safetensors"
    assert row["strength_model"] == 0.8
    rows = c.execute("SELECT COUNT(*) n FROM render_lora "
                     "WHERE strength_model BETWEEN 0.7 AND 0.9").fetchone()
    assert rows["n"] == 1
    c.close()


def test_rebuilding_does_not_duplicate(tmp_path):
    _write(tmp_path / "a.png", wf(lora_nodes=BUNDLE))
    db = tmp_path / "db.sqlite"
    build(tmp_path, db)
    build(tmp_path, db)
    c = connect(db)
    assert c.execute("SELECT COUNT(*) FROM render").fetchone()[0] == 1
    assert c.execute("SELECT COUNT(*) FROM render_lora").fetchone()[0] == 2
    c.close()


def test_an_unreadable_lora_stack_still_gets_a_row(tmp_path):
    """An index that drops a file is useless; one that stores a wrong value
    is worse. The row exists, the status says not to trust the empty list."""
    odd = {"30": {"class_type": "VendorLoraThing",
                  "inputs": {"lora_blob": "vendor.safetensors"}}}
    _write(tmp_path / "a.png", wf(lora_nodes=odd))
    db = tmp_path / "db.sqlite"
    build(tmp_path, db)
    c = connect(db)
    row = c.execute("SELECT lora_status FROM render").fetchone()
    assert row["lora_status"] == UNSUPPORTED
    c.close()


# ---------------------------------------------------------------- provenance

def test_model_identity_is_none_when_not_found(tmp_path):
    assert model_identity("nope.safetensors", [tmp_path]) is None


def test_model_identity_finds_a_real_file(tmp_path):
    (tmp_path / "anime_v3.safetensors").write_bytes(b"x" * 40)
    ident = model_identity("anime_v3.safetensors", [tmp_path])
    assert ident["size"] == 40


def test_provenance_reads_from_the_image_not_a_sidecar(tmp_path):
    p = _write(tmp_path / "a.png", wf(lora_nodes=NATIVE))
    (tmp_path / "anime_v3.safetensors").write_bytes(b"y" * 11)
    prov = provenance(p, [tmp_path])
    assert prov["model_identity"]["checkpoint"]["size"] == 11


# ---------------------------------------------------------------------- diff

def test_diff_reports_a_single_variable(tmp_path):
    a = _write(tmp_path / "a.png", wf(prompt="1girl"))
    b = _write(tmp_path / "b.png", wf(prompt="1girl, sunset"))
    d = compare_renders(a, b)
    assert d["prompt_changed"] and d["changed"] == 1
    assert d["single_variable"]


def test_diff_counts_a_seed_change(tmp_path):
    a = _write(tmp_path / "a.png", wf(seed=1))
    b = _write(tmp_path / "b.png", wf(seed=2))
    d = compare_renders(a, b)
    assert d["settings"]["seed"] == (1, 2)
    assert d["single_variable"]


def test_diff_sees_a_lora_strength_change(tmp_path):
    other = {"10": dict(NATIVE["10"])}
    other["10"]["inputs"] = dict(NATIVE["10"]["inputs"])
    other["10"]["inputs"]["strength_model"] = 0.2
    a = _write(tmp_path / "a.png", wf(lora_nodes=NATIVE))
    b = _write(tmp_path / "b.png", wf(lora_nodes=other))
    d = compare_renders(a, b)
    assert d["changed"] == 2          # removed at 0.8, added at 0.2
    assert not d["single_variable"]


def test_diff_will_not_call_it_single_variable_when_a_stack_is_unreadable(tmp_path):
    """Silence about what could not be read is how a comparison gets trusted
    further than it should be."""
    odd = {"30": {"class_type": "VendorLoraThing",
                  "inputs": {"lora_blob": "vendor.safetensors"}}}
    a = _write(tmp_path / "a.png", wf(prompt="1girl", lora_nodes=odd))
    b = _write(tmp_path / "b.png", wf(prompt="1girl, sunset", lora_nodes=odd))
    d = compare_renders(a, b)
    assert d["changed"] == 1
    assert not d["single_variable"]
    assert "could not be read" in d["caveat"]


def test_a_downstream_consumer_is_not_counted_as_unreadable():
    """A LoRA-manager graph has one node holding the stack and several
    consumers wired to it, each with its own empty {"__value__": []} widget.
    Counting those as unread reported every render partial and made the
    status column worthless -- 1,404 of 1,404 on a real corpus."""
    api = wf(lora_nodes={
        "20": BUNDLE["20"],
        "21": {"class_type": "Lora Loader (LoraManager)",
               "inputs": {"text": "", "loras": {"__value__": []},
                          "lora_stack": ["20", 0]}},
        "22": {"class_type": "TriggerWord Toggle (LoraManager)",
               "inputs": {"toggle_trigger_words": {"__value__": []},
                          "trigger_words": ["21", 2]}},
    })
    got, status = loras(api)
    assert status == OK
    assert len(got) == 2


def test_rgthree_power_lora_loader_is_read():
    api = wf(lora_nodes={"50": {
        "class_type": "Power Lora Loader (rgthree)",
        "inputs": {"PowerLoraLoaderHeaderWidget": {"type": "header"},
                   "lora_1": {"lora": "a.safetensors", "strength": 0.7,
                              "on": True},
                   "lora_2": {"lora": "b.safetensors", "strength": 1.0,
                              "on": False},
                   "model": ["4", 0]}}})
    got, status = loras(api)
    assert status == OK
    assert [(g["name"], g["strength_model"], g["active"]) for g in got] == [
        ("a.safetensors", 0.7, True), ("b.safetensors", 1.0, False)]


def test_an_empty_power_lora_loader_is_not_unreadable():
    """Observed in the wild: a Power Lora Loader placed but never filled in.
    It carries no LoRA, so it is not a LoRA this failed to read."""
    api = wf(lora_nodes={"51": {
        "class_type": "Power Lora Loader (rgthree)",
        "inputs": {"PowerLoraLoaderHeaderWidget": {"type": "header"},
                   "Add Lora": "", "model": ["4", 0]}}})
    assert loras(api)[1] == NONE


def test_a_trigger_word_payload_is_not_a_lora_payload():
    """A trigger-word toggle carries a populated __value__ too, but its
    entries are {"text": ...} -- words, not adapters. Counting them made 61
    renders in a real corpus claim a LoRA stack that could not be read."""
    api = wf(lora_nodes={
        "20": BUNDLE["20"],
        "22": {"class_type": "TriggerWord Toggle (LoraManager)",
               "inputs": {"toggle_trigger_words": {"__value__": [
                   {"text": "@style_a", "active": True, "strength": None}]},
                   "trigger_words": ["20", 2]}},
    })
    assert loras(api)[1] == OK


def test_inline_lora_syntax_is_read():
    """Found only by scanning a real corpus: 43 renders applied every LoRA
    as `<lora:name:0.8>` inside a text field, through a node with no
    filename, no __value__ and no lora_name. They were reported as having no
    LoRAs at all -- a silent false clean."""
    api = wf(lora_nodes={"60": {
        "class_type": "LoRA Text Loader (LoraManager)",
        "inputs": {"lora_syntax":
                   "<lora:style_x:0.8> <lora:style_y:1.00>"}}})
    got, status = loras(api)
    assert status == OK
    assert [(g["name"], g["strength_model"]) for g in got] == [
        ("style_x", 0.8), ("style_y", 1.0)]


def test_inline_syntax_without_a_strength_defaults_to_one():
    api = wf(lora_nodes={"60": {"class_type": "LoRA Text Loader",
                                "inputs": {"s": "<lora:plain>"}}})
    assert loras(api)[0][0]["strength_model"] == 1.0


def test_clip_strength_is_never_reported_as_model_strength():
    """Falling back to strengthTwo would invent a model strength out of a
    different quantity, indistinguishable downstream from a real reading."""
    api = wf(lora_nodes={"50": {
        "class_type": "Power Lora Loader (rgthree)",
        "inputs": {"lora_1": {"lora": "x.safetensors", "strengthTwo": 0.3,
                              "on": True}}}})
    got, status = loras(api)
    assert got[0]["strength_model"] is None
    assert got[0]["strength_clip"] == 0.3
    assert status == PARTIAL          # a missing strength is not "ok"


def test_an_adapter_not_spelled_lora_is_not_a_false_clean():
    """A LyCORIS/LoHa loader carries a real filename and a real strength."""
    api = wf(lora_nodes={"70": {
        "class_type": "LyCORISLoader",
        "inputs": {"lyco_name": "thing.safetensors", "strength_model": 0.5}}})
    assert loras(api)[1] == UNSUPPORTED


def test_a_checkpoint_on_a_lora_named_node_is_not_a_false_alarm():
    """`.safetensors` alone is not LoRA payload; the key has to name one."""
    api = wf(lora_nodes={"71": {
        "class_type": "LoraCompatCheckpointPicker",
        "inputs": {"ckpt_name": "base_model.safetensors"}}})
    assert loras(api)[1] == NONE


def test_a_strength_wired_in_as_a_link_is_not_silently_ok():
    api = wf(lora_nodes={"10": {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "x.safetensors",
                   "strength_model": ["99", 0], "strength_clip": 1.0}}})
    got, status = loras(api)
    assert got[0]["strength_model"] is None
    assert status == PARTIAL


def test_build_prunes_a_renamed_file(tmp_path):
    """A renamed file left its old row behind and every count was quietly
    too high."""
    a = _write(tmp_path / "a.png", wf(lora_nodes=NATIVE))
    db = tmp_path / "db.sqlite"
    build(tmp_path, db)
    a.rename(tmp_path / "a_renamed.png")
    build(tmp_path, db)
    c = connect(db)
    files = [r["file"] for r in c.execute("SELECT file FROM render")]
    assert files == ["a_renamed.png"]
    assert c.execute("SELECT COUNT(*) FROM render_lora").fetchone()[0] == 1
    c.close()


def test_build_does_not_prune_rows_belonging_to_another_root(tmp_path):
    """One database can index several folders. Pruning is scoped to the root
    being built, which is the only reason prune=True is safe as a default."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write(a / "one.png", wf())
    _write(b / "two.png", wf(seed=2))
    db = tmp_path / "db.sqlite"
    build(a, db)
    build(b, db)
    c = connect(db)
    files = sorted(r["file"] for r in c.execute("SELECT file FROM render"))
    assert files == ["one.png", "two.png"]
    c.close()


def test_build_prunes_a_deleted_file(tmp_path):
    _write(tmp_path / "a.png", wf())
    _write(tmp_path / "b.png", wf(seed=2))
    db = tmp_path / "db.sqlite"
    build(tmp_path, db)
    (tmp_path / "b.png").unlink()
    build(tmp_path, db)
    c = connect(db)
    assert c.execute("SELECT COUNT(*) FROM render").fetchone()[0] == 1
    c.close()


# ------------------------------------------------- regression on real shapes

def _real_cases():
    import pathlib
    f = pathlib.Path(__file__).parent / "fixtures" / "real_shapes.json"
    return json.loads(f.read_text(encoding="utf-8"))["cases"]


def test_real_workflow_fixtures_exist():
    """Synthetic fixtures prove a reader handles a shape it was written for.
    They cannot prove it does not misread the shapes that already worked --
    which is how three of the four lora_status corrections were needed. These
    graphs are what ComfyUI actually wrote, structure untouched, with only
    names and explicit words scrubbed."""
    cases = _real_cases()
    assert len(cases) >= 7
    shapes = {s for c in cases for s in c["expect_shapes"]}
    assert {"bundle", "inline_syntax"} <= shapes
    classes = {k for c in cases for k in c["lora_node_classes"]}
    assert "Power Lora Loader (rgthree)" in classes
    assert "LoRA Text Loader (LoraManager)" in classes


@pytest.mark.parametrize("case", _real_cases(),
                         ids=lambda c: c["source"])
def test_real_workflow_status_does_not_drift(case):
    """The regression the synthetic suite structurally cannot catch: a new
    reader that quietly reclassifies graphs that were already right."""
    got, status = loras(case["workflow"])
    assert status == case["expect_status"]
    assert len(got) == case["expect_lora_count"]
    assert sorted({e["shape"] for e in got}) == case["expect_shapes"]


@pytest.mark.parametrize("case", _real_cases(),
                         ids=lambda c: c["source"])
def test_real_workflows_scan_without_error(case):
    from renderdb.workflow import fingerprint, models, sampler_settings
    assert fingerprint(case["workflow"])
    sampler_settings(case["workflow"])
    models(case["workflow"])


def test_every_real_lora_row_has_a_strength():
    """The one column this package exists for."""
    for case in _real_cases():
        got, status = loras(case["workflow"])
        if status == OK:
            assert all(e["strength_model"] is not None for e in got)


# ------------------------------------------- the chunk layer real files have

def _realistic_png(text_chunks, idat_count=5, idat_size=2048):
    """A PNG shaped like the ones ComfyUI actually writes.

    Measured over 400 real renders: IHDR, one or two `tEXt` chunks keyed
    `prompt` and `workflow`, then several IDAT, then IEND -- 4 to 66 chunks a
    file, text payloads from 731 bytes to 124 KB. The fixtures elsewhere in
    this file start from a parsed graph and so never exercise the chunk
    walker at all; the older synthetic PNGs had no IDAT and one small text
    chunk, which is the easy case.

    No pixels: the IDAT payloads are filler. The point is the arithmetic that
    walks past them.
    """
    out = bytearray(b"\x89PNG\r\n\x1a\n")

    def put(ctype, payload):
        out.extend(struct.pack(">I", len(payload)))
        out.extend(ctype + payload)
        out.extend(struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))

    put(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    for key, value in text_chunks:
        put(b"tEXt", key.encode() + b"\x00" + value.encode())
    for i in range(idat_count):
        put(b"IDAT", bytes(idat_size))
    put(b"IEND", b"")
    return bytes(out)


def test_the_prompt_chunk_is_found_past_many_idat(tmp_path):
    p = tmp_path / "real.png"
    p.write_bytes(_realistic_png([("prompt", json.dumps(wf(lora_nodes=NATIVE)))],
                                 idat_count=30))
    row = scan_one(p)
    assert row["seed"] == 7
    assert row["loras"][0]["strength_model"] == 0.8


def test_workflow_chunk_present_but_prompt_wins(tmp_path):
    """Real renders carry both. `workflow` is the editor's copy and can
    differ from what ran; only `prompt` is authoritative."""
    editor = json.dumps({"nodes": [{"id": 1, "type": "Decoy"}]})
    p = tmp_path / "both.png"
    p.write_bytes(_realistic_png([
        ("workflow", editor),
        ("prompt", json.dumps(wf(seed=4242))),
    ]))
    assert scan_one(p)["seed"] == 4242


def test_a_large_text_chunk_parses(tmp_path):
    """Real payloads reach 124 KB; the synthetic ones elsewhere are bytes."""
    big = wf()
    big["pad"] = {"class_type": "CR Text", "inputs": {"text": "x" * 120000}}
    p = tmp_path / "big.png"
    p.write_bytes(_realistic_png([("prompt", json.dumps(big))]))
    assert len(read_workflow(p)) == len(big)


def test_a_render_with_no_text_chunk_is_skipped_not_crashed(tmp_path):
    p = tmp_path / "plain.png"
    p.write_bytes(_realistic_png([], idat_count=8))
    with pytest.raises(WorkflowError):
        scan_one(p)


# --------------------------------------------- a reader that fails is not "none"

@contextlib.contextmanager
def _registered(predicate, reader):
    workflow.LORA_READERS.insert(0, (predicate, reader))
    try:
        yield
    finally:
        workflow.LORA_READERS.pop(0)


def _boom(*a, **k):
    raise RuntimeError("this reader is broken")


def test_a_reader_that_raises_is_unreadable_not_absent():
    # A registered reader claimed the shape and then failed. Reporting
    # no_lora_nodes there states "there are none here" on the strength of a
    # crash -- the silent false clean this status column exists to prevent,
    # arriving through the documented extension point rather than through an
    # unknown shape.
    api = {"1": {"class_type": "StackHolder", "inputs": {"items": ["x"]}}}
    with _registered(lambda n: n.get("class_type") == "StackHolder", _boom):
        entries, status = workflow.loras(api)
    assert entries == []
    assert status == workflow.UNSUPPORTED


def test_a_predicate_that_raises_does_not_invent_a_lora_node():
    # A predicate that blows up says nothing about the node. Counting it as an
    # unreadable LoRA would report partial on graphs that hold none, which is
    # the first of the four corrections all over again.
    api = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x"}}}
    with _registered(_boom, _boom):
        entries, status = workflow.loras(api)
    assert entries == []
    assert status == workflow.NONE


def test_the_shipped_readers_raise_on_no_real_node():
    # The fix above only preserves the corpus numbers in the README if none of
    # the shipped readers currently crash. Locked in here rather than assumed.
    for case in _real_cases():
        for nid, node in case["workflow"].items():
            if not isinstance(node, dict):
                continue
            for predicate, reader in workflow.LORA_READERS:
                if predicate(node):
                    reader(nid, node)
