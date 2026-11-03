"""Check that every bits-per-byte figure quoted in the README's "The null arm, run" section
and in the site's null-arm paragraph is a number that exists in experiments/loadbearing/summary.json
(an arm mean, a seed value, a floor, a check value, or a difference between two of them).

    python scripts/check_loadbearing_text.py
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "experiments" / "loadbearing" / "summary.json").read_text())

values = set()
for exp in S["experiments"].values():
    for a in exp["arms"].values():
        values.add(a["bpb"]); values.update(a["values"])
    for d in exp.get("lr_check", {}).values():
        values.update(d.values())
    for d in exp.get("budget_check", {}).values():
        values.update(d.values())
    values.update(v for v in exp.get("floors", {}).values() if isinstance(v, float))
for d in S["floors"].get("at_budget", {}).values():
    values.update(v for v in d.values() if isinstance(v, float))
values.update(v for v in S["floors"].get("full_slice", {}).values() if isinstance(v, float))
diffs = {abs(a - b) for a, b in itertools.combinations(sorted(values), 2)}


def ok(x: float, text: str) -> bool:
    dec = len(text.split(".")[1]) if "." in text else 0
    tol = 0.5 * 10 ** -dec + 1e-9
    return any(abs(x - v) <= tol for v in values) or any(abs(x - v) <= tol for v in diffs)


def check(label: str, text: str) -> int:
    bad = []
    for m in re.finditer(r"(?<![\w.%])(\d\.\d{2,3})(?![\d%])", text):
        if not ok(float(m.group(1)), m.group(1)):
            bad.append(m.group(1))
    print(f"{label}: {len(bad)} unmatched {sorted(set(bad))}")
    return len(bad)


readme = (ROOT / "README.md").read_text()
sec = readme[readme.index("## The null arm, run"):readme.index("## The one habit")]
n = check("README section", sec)
site = (ROOT / "site" / "index.template.html").read_text()
nm = site[site.index('<section id="nullarm">'):site.index('<section id="standard">')]
n += check("site section", nm)
sys.exit(1 if n else 0)
