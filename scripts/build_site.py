"""Build docs/index.html from site/index.template.html and the data files.

    python scripts/build_site.py

The template holds the page; this script injects the data (projects, scores,
curves, the inert-mechanism table, the component-share table) as a JSON object
and wraps the result in a document skeleton.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIR_TO_ID = {
    "developmental": "dpnca", "NCALM": "ncalm",
    "active_inference_thermodynamic_computation_llm": "wavelm", "optical_model": "sfnn",
    "personal-website": "pw", "HDC_LM": "hdclm", "mashi": "mashi", "antientrhopy": "nen",
    "sampling_enthropic": "fedn", "olento": "olento", "new_model_2026": "harm",
    "evolve-hierarchical_nca": "wave-treatise", "principled_model": "apeiron",
    "intelligence_cant_be_modeled_it_must_be_grown": "drn", "rglm_lineage": "rglm",
    "chaotic_lineage": "chaotic",
}

SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="An audit of sixteen post-transformer language-model architectures, 2024–2026: method, evidence grading, what survived, the ranked table, and the cross-cutting finding.">
<meta property="og:title" content="Post-Transformer Research">
<meta property="og:description" content="Sixteen architectures, one standard, no winners. What twenty months of attention-free and backprop-free language-model research measured.">
<meta property="og:type" content="article">
<meta property="og:image" content="https://peterpetu.github.io/post-transformer-research/figures/ranking.png">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)} img{max-width:100%} [hidden]{display:none!important}</style>
__HEAD__
</head>
<body>
__BODY__
</body>
</html>
"""


def main() -> None:
    scores = {}
    with open(ROOT / "data" / "scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            pid = DIR_TO_ID[row["project"]]
            scores[pid] = {k: (float(v) if k not in ("project", "codename") else v) for k, v in row.items()}
    data = {
        "projects": json.load(open(ROOT / "data" / "projects.json", encoding="utf-8")),
        "scores": scores,
        "inert": json.load(open(ROOT / "data" / "inert.json", encoding="utf-8")),
        "share": json.load(open(ROOT / "data" / "component_share.json", encoding="utf-8")),
    }
    data.update(json.load(open(ROOT / "data" / "curves.json", encoding="utf-8")))
    lb = ROOT / "experiments" / "loadbearing" / "summary.json"
    data["loadbearing"] = json.load(open(lb, encoding="utf-8")) if lb.exists() else None
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    template = (ROOT / "site" / "index.template.html").read_text(encoding="utf-8")
    page = template.replace("__DATA__", payload)

    # The artifact preview takes the bare page; GitHub Pages needs a full document.
    (ROOT / "site" / "preview.html").write_text(page, encoding="utf-8")
    head_end = page.index("</style>") + len("</style>")
    head, body = page[:head_end], page[head_end:]
    doc = SKELETON.replace("__HEAD__", head).replace("__BODY__", body)
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("", encoding="utf-8")
    # the social-preview image the page's meta tag points at
    import shutil
    (ROOT / "docs" / "figures").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "figures" / "ranking.png", ROOT / "docs" / "figures" / "ranking.png")
    print(f"wrote {out} ({out.stat().st_size:,} bytes) and site/preview.html")


if __name__ == "__main__":
    main()
