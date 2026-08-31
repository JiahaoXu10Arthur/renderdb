"""Command line entry point.

    renderdb build <dir> [--db renders.db]
    renderdb why <image.png> [--models <dir> ...]
    renderdb diff <a.png> <b.png>
    renderdb stats [--db renders.db]
    renderdb sql "SELECT ..." [--db renders.db]

Exit 0 on success, 3 when the command could not run.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from . import build, compare_renders, connect, provenance
from .workflow import NONE, OK, PARTIAL, UNSUPPORTED, WorkflowError

DEFAULT_DB = "renders.db"


def _fail(msg: str, *extra: str) -> int:
    print("error: %s" % msg, file=sys.stderr)
    for line in extra:
        print("       %s" % line, file=sys.stderr)
    return 3


def _opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args and \
        args.index(name) + 1 < len(args) else default


def _cmd_build(args) -> int:
    if not args:
        return _fail("build needs a directory")
    root = Path(args[0])
    if not root.is_dir():
        return _fail("%s is not a directory" % root)
    db = _opt(args, "--db", DEFAULT_DB)
    reasons = {}

    def note(png, err):
        key = str(err).split(" (")[0][:70]
        reasons[key] = reasons.get(key, 0) + 1

    n, skipped = build(root, db, on_error=note)
    print("indexed %d, skipped %d -> %s" % (n, skipped, db))
    for why, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
        print("  skipped %5d  %s" % (count, why))
    return 0


def _cmd_why(args) -> int:
    if not args:
        return _fail("why needs an image")
    paths = []
    i = 1
    while i < len(args):
        if args[i] == "--models" and i + 1 < len(args):
            paths.append(args[i + 1])
            i += 2
        else:
            i += 1
    try:
        p = provenance(args[0], paths)
    except WorkflowError as e:
        return _fail(str(e))
    except OSError as e:
        return _fail(str(e))

    print(p["file"])
    print("  pipeline   %s" % p["fingerprint"])
    for key in ("checkpoint", "unet", "vae"):
        if p.get(key):
            ident = (p.get("model_identity") or {}).get(key)
            extra = ""
            if ident:
                extra = "  (%d bytes)" % ident["size"]
            elif paths:
                extra = "  (not found on disk)"
            print("  %-10s %s%s" % (key, p[key], extra))
    bits = [("seed", p["seed"]), ("steps", p["steps"]), ("cfg", p["cfg"]),
            ("sampler", p["sampler"]), ("scheduler", p["scheduler"]),
            ("denoise", p["denoise"])]
    print("  settings   %s" % ", ".join(
        "%s=%s" % (k, "?" if v is None else v) for k, v in bits))

    status = p["lora_status"]
    if status == NONE:
        print("  loras      none on this graph")
    elif status == UNSUPPORTED:
        print("  loras      COULD NOT READ -- this graph has LoRA nodes in a "
              "shape with no reader.")
        print("             An empty list here means 'not read', not 'none'.")
    else:
        note = "  (partial: some nodes unreadable)" if status == PARTIAL else ""
        print("  loras%s" % note)
        for e in p["loras"]:
            sm = "?" if e["strength_model"] is None else e["strength_model"]
            print("    %-46s %-6s %s" % (e["name"][:46], sm,
                                         "" if e["active"] else "(off)"))
    if p.get("prompt"):
        print("  prompt     %s" % p["prompt"][:110])
        print("             (best effort -- text a LoRA manager or upsampler "
              "adds at run time is not in the file)")
    return 0


def _cmd_diff(args) -> int:
    if len(args) < 2:
        return _fail("diff needs two images")
    try:
        d = compare_renders(args[0], args[1])
    except (WorkflowError, OSError) as e:
        return _fail(str(e))
    for key, (x, y) in sorted(d["settings"].items()):
        print("  %-12s %s  ->  %s" % (key, x, y))
    for kind, name, strength in d["loras"]:
        print("  lora %-7s %s @ %s" % (kind, name, strength))
    if d["prompt_changed"]:
        print("  prompt       changed")
    print()
    print("  %d change%s%s" % (d["changed"], "" if d["changed"] == 1 else "s",
                               "  (single variable)" if d["single_variable"]
                               else ""))
    if d["caveat"]:
        print("  ! %s" % d["caveat"])
    return 0


def _cmd_stats(args) -> int:
    db = _opt(args, "--db", DEFAULT_DB)
    if not Path(db).exists():
        return _fail("%s does not exist -- run `renderdb build` first" % db)
    c = connect(db)
    n = c.execute("SELECT COUNT(*) FROM render").fetchone()[0]
    print("%d renders" % n)
    for row in c.execute("SELECT lora_status, COUNT(*) n FROM render "
                         "GROUP BY lora_status ORDER BY n DESC"):
        print("  lora %-22s %d" % (row["lora_status"], row["n"]))
    top = c.execute(
        "SELECT name, COUNT(*) n, MIN(strength_model) lo, "
        "MAX(strength_model) hi FROM render_lora WHERE active = 1 "
        "GROUP BY name ORDER BY n DESC LIMIT 8").fetchall()
    if top:
        print("  most used loras:")
        for r in top:
            span = "%s" % r["lo"] if r["lo"] == r["hi"] \
                else "%s-%s" % (r["lo"], r["hi"])
            print("    %-42s %4d  strength %s" % (r["name"][:42], r["n"], span))
    c.close()
    return 0


def _cmd_sql(args) -> int:
    if not args:
        return _fail("sql needs a query")
    db = _opt(args, "--db", DEFAULT_DB)
    if not Path(db).exists():
        return _fail("%s does not exist -- run `renderdb build` first" % db)
    c = connect(db)
    try:
        rows = c.execute(args[0]).fetchall()
    except sqlite3.Error as e:
        c.close()
        return _fail(str(e))
    for r in rows[:200]:
        print("  " + " | ".join(str(x) for x in tuple(r)))
    print("  (%d rows)" % len(rows))
    c.close()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    handlers = {"build": _cmd_build, "why": _cmd_why, "diff": _cmd_diff,
                "stats": _cmd_stats, "sql": _cmd_sql}
    if argv[0] not in handlers:
        return _fail("unknown command %r" % argv[0], "try --help")
    return handlers[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
