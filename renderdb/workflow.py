"""Read a ComfyUI PNG's embedded workflow, and pull structured facts out of it.

ComfyUI writes the API graph it actually ran into every PNG, so a folder of
renders is already a complete experiment record. It is just not queryable.

Node shapes, not node ids
-------------------------
The script this grew from located things by literal node number -- ``1043`` is
the quality prefix, ``1130`` is the handwritten prompt. That works until
ComfyUI renumbers the graph on the next save, and it never worked on anyone
else's workflow. Nothing here matches on a node id.

What it matches on instead is a **shape**: a set of input keys that a family of
nodes uses. ``LoraLoader`` has ``lora_name`` and ``strength_model`` as flat
values; LoRA-manager nodes bundle a whole stack into one widget as a list of
dicts under ``__value__``. Those are two shapes, and each needs its own reader.

Getting that wrong is not hypothetical. The original only understood the
bundle shape, because that is what its author's workflow used -- across 1,603
of their renders there is not one native ``LoraLoader`` node, so nothing ever
signalled the gap. Anyone with a stock ComfyUI setup would have got LoRA names
with no strengths and no error, which is exactly the field this package exists
to make queryable.

The registry below is the fix and the extension point: add a reader, get a
shape. An unrecognised shape is recorded as unresolved, never guessed at.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ["read_workflow", "read_text_chunks", "loras", "sampler_settings",
           "models", "fingerprint", "prompt_text", "WorkflowError",
           "LORA_READERS", "OK", "PARTIAL", "UNSUPPORTED", "NONE"]

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

#: Every LoRA on the graph was read, name and strength.
OK = "ok"
#: Some were read; at least one node held a shape with no reader.
PARTIAL = "partial"
#: LoRA-ish nodes are present and none could be read. An empty LoRA list for
#: this render means "not read", not "no LoRAs" -- the difference between no
#: rows and zero rows meaning something.
UNSUPPORTED = "unsupported_node_shape"
#: No LoRA nodes on the graph at all. An empty list here is a fact.
NONE = "no_lora_nodes"

_SAMPLERS = ("KSampler", "SamplerCustom", "KSamplerAdvanced")

_SAMPLER_FIELDS = ("seed", "noise_seed", "steps", "cfg", "sampler_name",
                   "scheduler", "denoise")

_MODEL_KEYS = ("ckpt_name", "unet_name", "vae_name", "clip_name")

_TRAVERSABLE = ("CLIPTextEncode", "TextEncode", "StringConcatenate",
                "JoinStringMulti", "JoinString", "PrimitiveString",
                "StringLiteral", "CR Text", "Text Multiline", "ShowText",
                "StringConstantMultiline")

_SKIP_KEYS = frozenset({
    "delimiter", "separator", "type", "device", "clip", "vae", "model",
    "ckpt_name", "lora_name", "clip_name", "unet_name", "sampler_name",
    "scheduler", "filename_prefix", "image", "latent_image", "weight_dtype",
})

#: Words that mark a node as applying some kind of weight adapter. Used only
#: to decide whether an unread node is worth reporting as unreadable.
_ADAPTER_WORDS = ("lora", "lycoris", "loha", "lokr", "locon", "dora", "lyco")

_MAX_DEPTH = 24


class WorkflowError(Exception):
    """The PNG carried no readable ComfyUI workflow."""


def read_text_chunks(png) -> Dict[str, str]:
    """Every text chunk in a PNG, without Pillow.

    A ``tEXt`` chunk is a length, a four-byte type, ``keyword\\0text``, and a
    CRC. Reading it by hand keeps this package importable anywhere, which
    matters more for an indexer that might run in a build step than the forty
    lines cost.
    """
    data = Path(png).read_bytes()
    if not data.startswith(_PNG_SIG):
        raise WorkflowError("%s is not a PNG" % Path(png).name)
    out: Dict[str, str] = {}
    pos = len(_PNG_SIG)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IEND":
            break
        try:
            if ctype == b"tEXt":
                k, _, v = body.partition(b"\x00")
                out[k.decode("latin-1")] = v.decode("utf-8", "replace")
            elif ctype == b"zTXt":
                k, _, rest = body.partition(b"\x00")
                out[k.decode("latin-1")] = zlib.decompress(
                    rest[1:]).decode("utf-8", "replace")
        except Exception:
            continue
    return out


def read_workflow(png) -> dict:
    """The API graph, which is what ran. Not the editor's copy."""
    chunks = read_text_chunks(png)
    raw = chunks.get("prompt")
    if not raw:
        have = ", ".join(sorted(chunks)) or "none"
        raise WorkflowError("%s has no embedded workflow (chunks: %s)"
                            % (Path(png).name, have))
    try:
        api = json.loads(raw)
    except ValueError as e:
        raise WorkflowError("%s: unreadable workflow (%s)"
                            % (Path(png).name, e))
    if not isinstance(api, dict) or not api:
        raise WorkflowError("%s: empty workflow" % Path(png).name)
    return api


# --------------------------------------------------------------- LoRA shapes

def _read_native_loader(nid: str, node: dict) -> List[dict]:
    """Stock ``LoraLoader`` / ``LoraLoaderModelOnly``: flat scalar inputs."""
    ins = node.get("inputs") or {}
    name = ins.get("lora_name")
    if not isinstance(name, str) or not name:
        return []
    sm = ins.get("strength_model", ins.get("strength"))
    sc = ins.get("strength_clip")
    return [{
        "name": name,
        "strength_model": sm if isinstance(sm, (int, float)) else None,
        "strength_clip": sc if isinstance(sc, (int, float)) else None,
        "active": True,
        "shape": "native_loader",
        "node": nid,
    }]


def _read_bundle(nid: str, node: dict) -> List[dict]:
    """LoRA-manager style: a whole stack inside one widget as ``__value__``."""
    out = []
    for val in (node.get("inputs") or {}).values():
        if not isinstance(val, dict):
            continue
        for item in (val.get("__value__") or []):
            if not isinstance(item, dict) or not isinstance(item.get("name"),
                                                            str):
                continue
            sm = item.get("strength", item.get("model_strength"))
            sc = item.get("clipStrength", item.get("clip_strength"))
            out.append({
                "name": item["name"],
                "strength_model": sm if isinstance(sm, (int, float)) else None,
                "strength_clip": sc if isinstance(sc, (int, float)) else None,
                "active": bool(item.get("active", True)),
                "shape": "bundle",
                "node": nid,
            })
    return out


#: ``<lora:name:0.8>`` -- A1111 prompt syntax, which several ComfyUI nodes
#: accept as a plain string field instead of structured inputs.
_INLINE = re.compile(r"<lora:([^:>]+)(?::([-\d.]+))?(?::([-\d.]+))?>",
                     re.IGNORECASE)


def _read_inline(nid: str, node: dict) -> List[dict]:
    """LoRAs written as ``<lora:name:strength>`` inside a text field.

    Found only by scanning a real corpus: 43 renders applied every one of
    their LoRAs this way, through a node whose inputs hold no filename, no
    ``__value__`` and no ``lora_name``. They were reported as having no LoRAs
    at all -- a silent false clean, which is the one failure this module is
    supposed to make impossible.
    """
    out = []
    for val in (node.get("inputs") or {}).values():
        if not isinstance(val, str) or "<lora:" not in val.lower():
            continue
        for name, sm, sc in _INLINE.findall(val):
            try:
                model = float(sm) if sm else 1.0
            except ValueError:
                model = None
            try:
                clip = float(sc) if sc else None
            except ValueError:
                clip = None
            out.append({
                "name": name.strip(),
                "strength_model": model,
                "strength_clip": clip,
                "active": True,
                "shape": "inline_syntax",
                "node": nid,
            })
    return out


def _is_inline(node: dict) -> bool:
    return any(isinstance(v, str) and "<lora:" in v.lower()
               for v in (node.get("inputs") or {}).values())


def _is_native(node: dict) -> bool:
    ins = node.get("inputs") or {}
    return isinstance(ins.get("lora_name"), str)


def _read_power_loader(nid: str, node: dict) -> List[dict]:
    """rgthree Power Lora Loader: one dict per slot, ``lora_1``, ``lora_2``..."""
    out = []
    for key, val in sorted((node.get("inputs") or {}).items()):
        if not isinstance(val, dict):
            continue
        name = val.get("lora")
        if not isinstance(name, str) or not name:
            continue
        # Only ``strength`` is the model strength. Falling back to
        # ``strengthTwo`` would report the *clip* strength as the model
        # strength -- a different quantity, invented, and indistinguishable
        # from a real reading downstream.
        sm = val.get("strength")
        sc = val.get("strengthTwo", val.get("strengthClip"))
        out.append({
            "name": name,
            "strength_model": sm if isinstance(sm, (int, float)) else None,
            "strength_clip": sc if isinstance(sc, (int, float)) else None,
            "active": bool(val.get("on", True)),
            "shape": "power_loader",
            "node": nid,
        })
    return out


def _is_power_loader(node: dict) -> bool:
    return any(isinstance(v, dict) and isinstance(v.get("lora"), str)
               for v in (node.get("inputs") or {}).values())


def _is_bundle(node: dict) -> bool:
    return any(isinstance(v, dict) and "__value__" in v
               for v in (node.get("inputs") or {}).values())


def _carries_lora_data(node: dict) -> bool:
    """Does this node actually hold LoRA data, or is it downstream plumbing?

    A LoRA-manager graph has one node holding the stack and several
    *consumers* wired to it -- a loader, a trigger-word toggle -- each with
    its own empty ``{"__value__": []}`` widget and a link to the real source.
    Counting those as "LoRA nodes I could not read" reports every render as
    partially unreadable and makes the status column worthless. The question
    is whether there is payload here, not whether the class name says lora.
    """
    for val in (node.get("inputs") or {}).values():
        if isinstance(val, dict):
            # The payload has to look like *LoRA* entries. A trigger-word
            # toggle on the same graph also carries a populated ``__value__``,
            # but its entries are ``{"text": ...}`` -- words, not adapters.
            # Counting those made 61 renders in a real corpus report a LoRA
            # stack this could not read, when there was none to read.
            for item in (val.get("__value__") or []):
                if isinstance(item, dict) and (
                        isinstance(item.get("name"), str)
                        or isinstance(item.get("lora"), str)):
                    return True
            if isinstance(val.get("lora"), str) and val["lora"]:
                return True
    for key, val in (node.get("inputs") or {}).items():
        if not isinstance(val, str):
            continue
        if "<lora:" in val.lower():
            return True
        # A checkpoint filename sitting on a node whose class happens to
        # contain "lora" is not LoRA payload. Scope the filename test to keys
        # that name an adapter.
        if val.lower().endswith(".safetensors") and                 any(w in str(key).lower() for w in _ADAPTER_WORDS):
            return True
    return isinstance((node.get("inputs") or {}).get("lora_name"), str)


#: ``(predicate, reader)`` pairs. Add one to support a node family. Order is
#: not significant -- the first matching predicate wins, and a node matching
#: none is what ``UNSUPPORTED`` reports.
LORA_READERS: List[Tuple] = [
    (_is_native, _read_native_loader),
    (_is_power_loader, _read_power_loader),
    (_is_bundle, _read_bundle),
    (_is_inline, _read_inline),
]


def loras(api: dict) -> Tuple[List[dict], str]:
    """Every LoRA on the graph, as ``(entries, status)``.

    The status is the honest part. An empty list with ``UNSUPPORTED`` means
    "there are LoRA nodes here and I could not read them", which is a very
    different fact from ``NONE``.
    """
    found: List[dict] = []
    looks_lora = unreadable = 0
    for nid in sorted(api):
        node = api.get(nid)
        if not isinstance(node, dict):
            continue

        # Ask the registry first. A pre-filter that decides what counts as a
        # LoRA node before consulting the readers would gatekeep every reader
        # anyone adds -- the extension point would look present and never
        # fire.
        got: List[dict] = []
        failed = False
        for predicate, reader in LORA_READERS:
            try:
                matched = predicate(node)
            except Exception:
                # A predicate that blows up says nothing about this node.
                # Counting it as an unreadable LoRA would report partial on
                # graphs holding none, which is correction 1 all over again.
                continue
            if not matched:
                continue
            try:
                got = reader(nid, node) or []
            except Exception:
                # A reader claimed this shape and then crashed. That is a LoRA
                # node this build cannot see, not the absence of one, and
                # LORA_READERS is a documented extension point -- assume the
                # readers are incomplete *and* that they break.
                failed = True
                got = []
            if got:
                break
        if got:
            found.extend(got)
            continue
        if failed:
            # Only a raise counts. A reader returning nothing may have found
            # an empty node, which carries no LoRA -- that is correction 3,
            # and calling it unreadable would undo it.
            looks_lora += 1
            unreadable += 1
            continue

        # Nothing read it. Did it nonetheless look like a LoRA node? That is
        # the difference between "no LoRAs here" and "LoRAs I cannot see".
        cls = str(node.get("class_type") or "")
        ins = node.get("inputs") or {}
        # Adapters are not all spelled "lora". A LyCORIS / LoHa / LoCon
        # loader carries a real filename and a real strength and would
        # otherwise be reported as "no LoRA nodes here" -- a false clean.
        low = cls.lower()
        lora_ish = (any(w in low for w in _ADAPTER_WORDS)
                    or any(any(w in str(k).lower() for w in _ADAPTER_WORDS)
                           for k in ins))
        if lora_ish and _carries_lora_data(node):
            looks_lora += 1
            unreadable += 1

    looks_lora += len(found)
    if not looks_lora:
        return [], NONE
    # A LoRA whose strength was read as nothing is not fully read. Reporting
    # OK there would say "name and strength recovered" about a row whose
    # strength column is NULL, and the whole point of this package is that
    # column.
    incomplete = any(e["strength_model"] is None for e in found)
    if found and not unreadable and not incomplete:
        return found, OK
    if found:
        return found, PARTIAL
    return [], UNSUPPORTED


# ------------------------------------------------------------ other structure

def _scalar(api: dict, value, depth: int = 0):
    """A sampler setting, following one link if it is wired in.

    Seeds are usually not literals -- they come from a seed node. Reading only
    literals would silently miss the most important controlled variable.
    """
    if depth > 4:
        return None
    if not isinstance(value, list):
        return value
    if len(value) != 2:
        return None
    node = api.get(str(value[0]))
    if not isinstance(node, dict):
        return None
    ins = node.get("inputs") or {}
    numbers = [v for v in ins.values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    links = [v for v in ins.values() if isinstance(v, list)]
    if len(numbers) == 1:
        return numbers[0]
    if len(links) == 1:
        return _scalar(api, links[0], depth + 1)
    return None


def sampler_settings(api: dict) -> Dict[str, object]:
    """Seed, steps, cfg, sampler, scheduler, denoise -- ``None`` when unread."""
    out: Dict[str, object] = {}
    for nid in sorted(api):
        node = api.get(nid) or {}
        if not any(s in str(node.get("class_type") or "") for s in _SAMPLERS):
            continue
        ins = node.get("inputs") or {}
        for key in _SAMPLER_FIELDS:
            if key in ins and key not in out:
                out[key] = _scalar(api, ins[key])
    if "seed" not in out and "noise_seed" in out:
        out["seed"] = out["noise_seed"]
    return out


def models(api: dict) -> Dict[str, str]:
    """Checkpoint / unet / vae / clip filenames, by input key."""
    out: Dict[str, str] = {}
    for nid in sorted(api):
        ins = (api.get(nid) or {}).get("inputs") or {}
        for key in _MODEL_KEYS:
            v = ins.get(key)
            if isinstance(v, str) and v and key not in out:
                out[key] = v
    return out


def fingerprint(api: dict) -> str:
    """A stable id for the *pipeline*, ignoring what was typed into it.

    Node classes and how they are wired, not their widget values. Two renders
    with the same fingerprint went through the same machine; that is what makes
    "did I change the pipeline or the prompt" answerable later.
    """
    edges = []
    for nid in sorted(api, key=lambda x: str(x)):
        node = api.get(nid) or {}
        cls = str(node.get("class_type") or "")
        for key, val in sorted((node.get("inputs") or {}).items()):
            if isinstance(val, list) and len(val) == 2:
                src = str((api.get(str(val[0])) or {}).get("class_type") or "")
                edges.append("%s.%s<-%s" % (cls, key, src))
            else:
                edges.append("%s.%s" % (cls, key))
    return hashlib.sha256("\n".join(sorted(edges)).encode()).hexdigest()[:16]


def _resolve(api: dict, value, depth: int = 0) -> List[str]:
    if depth > _MAX_DEPTH:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and len(value) == 2:
        node = api.get(str(value[0]))
        if not isinstance(node, dict):
            return []
        if not any(s in str(node.get("class_type") or "")
                   for s in _TRAVERSABLE):
            return []
        out: List[str] = []
        for k, v in (node.get("inputs") or {}).items():
            if k in _SKIP_KEYS or isinstance(v, (int, float, bool)):
                continue
            out.extend(_resolve(api, v, depth + 1))
        return out
    return []


def prompt_text(api: dict, polarity: str = "positive") -> Optional[str]:
    """Best-effort prompt text, for full-text search only.

    Deliberately not a structured column. This walk cannot see text a LoRA
    manager or an upsampler produces at run time, so what it returns may be a
    fraction of what the model received. Good enough to search; not good
    enough to compare on, and the schema keeps it out of the exact-match
    columns for that reason.
    """
    for nid in sorted(api):
        node = api.get(nid) or {}
        if not any(s in str(node.get("class_type") or "") for s in _SAMPLERS):
            continue
        ref = (node.get("inputs") or {}).get(polarity)
        if ref is None:
            continue
        text = ", ".join(p for p in _resolve(api, ref) if p.strip())
        if text.strip():
            return text
    return None
