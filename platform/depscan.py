"""
platform.depscan -- distributed dependency annotation scanner.

Each module carries its own DEPENDENCIES block next to the code. The scanner
parses these with the AST (no imports, so it is safe), checks them against the
running interpreter, and prints a manifest.
Run:  python -m platform.depscan
"""
from __future__ import annotations

import ast
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version

from .backends import backend_name, formal_core, has_numpy


def _vtuple(v):
    parts = re.split(r"[.+\-]", str(v))
    out = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p)
    return out


def _cmp(installed: str, spec: str) -> bool:
    spec = spec.strip()
    if spec in ("*", "", "any"):
        return True
    m = re.match(r"(~=|==|>=|<=|>|<|=)\s*([0-9][0-9A-Za-z.\-]*)", spec)
    if not m:
        return True
    op, want = m.group(1), m.group(2)
    a, b = _vtuple(installed), _vtuple(want)
    pad = max(len(a), len(b))
    a += [0] * (pad - len(a)); b += [0] * (pad - len(b))
    if op == "~=":
        return a >= b and a[:len(_vtuple(want)) - 1] == b[:len(_vtuple(want)) - 1]
    return {"==": a == b, "=": a == b, ">=": a >= b, "<=": a <= b,
            ">": a > b, "<": a < b}[op]


def _installed(dist: str):
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def read_annotations(path: str):
    try:
        tree = ast.parse(open(path, "r", encoding="utf-8").read())
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DEPENDENCIES":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return None
    return None


def _backend_ok(name: str) -> bool:
    return {"formal": formal_core() is not None,
            "numpy": has_numpy(), "python": True}.get(name, False)


def scan(root: str | None = None):
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for dirpath, dirs, files in os.walk(root):
        if os.path.basename(dirpath) in ("out_platform", "__pycache__", "archive"):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            ann = read_annotations(path)
            if ann is None:
                continue
            rel = os.path.relpath(path, root)
            deps = []
            for dist, spec in ann.get("requires", {}).items():
                got = _installed(dist)
                deps.append((dist, spec, got, got is not None and _cmp(got, spec)))
            opt = []
            for dist, spec in ann.get("optional", {}).items():
                got = _installed(dist)
                opt.append((dist, got is not None))
            backs = [(b, _backend_ok(b)) for b in ann.get("backends", [])]
            rows.append({"file": rel, "status": ann.get("status", "?"),
                         "targets": ann.get("targets", ""), "requires": deps,
                         "optional": opt, "backends": backs,
                         "runnable": all(d[3] for d in deps)
                         and any(b[1] for b in backs)})
    rows.sort(key=lambda r: r["file"])
    return rows


def format_report(rows):
    lines = ["dependency manifest  (interpreter %s, measurement backend %s)"
             % (sys.version.split()[0], backend_name()), "=" * 72]
    for r in rows:
        flag = "OK " if r["runnable"] else "-- "
        lines.append("%s%-44s [%s] %s" % (flag, r["file"], r["status"],
                                          r["targets"]))
        for dist, spec, got, ok in r["requires"]:
            lines.append("      req %-16s %-8s installed %-10s %s"
                         % (dist, spec, got or "MISSING",
                            "yes" if ok else "NO"))
        for dist, present in r["optional"]:
            lines.append("      opt %-16s %s" % (dist,
                         "present" if present else "absent"))
        if r["backends"]:
            lines.append("      backends " + ", ".join(
                ("%s:%s" % (b, "yes" if ok else "no")) for b, ok in r["backends"]))
    runnable = sum(1 for r in rows if r["runnable"])
    lines.append("=" * 72)
    lines.append("%d annotated modules, %d runnable in this interpreter"
                 % (len(rows), runnable))
    return "\n".join(lines)


def _main():
    rows = scan()
    print(format_report(rows))


if __name__ == "__main__":
    _main()
