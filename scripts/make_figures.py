"""Render the figures in README.md from data/scores.csv and data/projects.json.

    python scripts/make_figures.py

Writes PNG and SVG to figures/.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3, LINE = "#0b0b0b", "#52514e", "#898781", "#d9d7d2"
SURFACE = "#fcfcfb"

OUTCOME_COLOUR = {
    "survived": BLUE,
    "current": AQUA,
    "no-fair-baseline": ORANGE,
    "no-artifacts": ORANGE,
    "lost-or-equivalent": "#f3a37d",
    "did-not-learn": INK3,
    "unbuilt": "#d6d4cf",
}
OUTCOME_LABEL = {
    "survived": "measured against a fair baseline",
    "no-fair-baseline": "measured, no fair baseline or no surviving log",
    "no-artifacts": "measured, no fair baseline or no surviving log",
    "lost-or-equivalent": "lost to, or equivalent to, a known method",
    "did-not-learn": "never learned",
    "unbuilt": "unbuilt",
    "current": "after the audit (not scored)",
}
THREAD_COLOUR = {
    "grown": BLUE,
    "physics": ORANGE,
    "active-inference": "#8e5bd6",
    "local-learning": AQUA,
    "current": INK2,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load():
    scores = {}
    with open(ROOT / "data" / "scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[row["project"]] = row
    projects = json.load(open(ROOT / "data" / "projects.json"))
    return scores, projects


# scores.csv keys its rows by source directory; map them to projects.json ids.
DIR_TO_ID = {
    "developmental": "dpnca", "NCALM": "ncalm",
    "active_inference_thermodynamic_computation_llm": "wavelm", "optical_model": "sfnn",
    "personal-website": "pw", "HDC_LM": "hdclm", "mashi": "mashi", "antientrhopy": "nen",
    "sampling_enthropic": "fedn", "olento": "olento", "new_model_2026": "harm",
    "evolve-hierarchical_nca": "wave-treatise", "principled_model": "apeiron",
    "intelligence_cant_be_modeled_it_must_be_grown": "drn", "rglm_lineage": "rglm",
    "chaotic_lineage": "chaotic",
}
SHORT = {
    "dpnca": "DPNCA-LM", "ncalm": "NCALM", "wavelm": "WaveLM", "hdclm": "HDC-LM", "sfnn": "SFNN (plan)",
    "pw": "Mamba+DeltaNet (V7/V10.1)", "mashi": "APCN-LM / VTM-LM", "nen": "NEN", "fedn": "FEDN / CAIN",
    "olento": "olento (EDO)", "harm": "HARM-LM", "wave-treatise": "Wave-AI treatise", "apeiron": "APEIRON",
    "drn": "DRN", "rglm": "RGLM lineage", "chaotic": "Chaotic Reservoir LM",
    "evolve": "developmental-lm", "fly": "FlyBrainLLM",
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def fig_ranking(scores, projects):
    by_id = {p["id"]: p for p in projects}
    rows = []
    for d, s in scores.items():
        pid = DIR_TO_ID[d]
        rows.append((SHORT[pid], float(s["composite"]), by_id[pid]["outcome"]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    y = range(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[OUTCOME_COLOUR[r[2]] for r in rows], height=0.72)
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.05, i, f"{r[1]:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 5.3)
    ax.set_xlabel("composite score (mean of 15 facets, each 1–10)")
    ax.set_title("Sixteen programmes, ranked by the audit's composite score", loc="left", fontsize=12, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    handles = [Patch(color=OUTCOME_COLOUR[k], label=OUTCOME_LABEL[k])
               for k in ("survived", "no-fair-baseline", "lost-or-equivalent", "did-not-learn", "unbuilt")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.11), ncol=2, frameon=False, fontsize=8.5)
    save(fig, "ranking")


def fig_idea_vs_evidence(scores, projects):
    by_id = {p["id"]: p for p in projects}
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    xs, ys = [], []
    for d, s in scores.items():
        pid = DIR_TO_ID[d]
        x, yv = float(s["novelty"]), float(s["performance"])
        xs.append(x); ys.append(yv)
        ax.scatter(x, yv, s=110, color=OUTCOME_COLOUR[by_id[pid]["outcome"]], edgecolor=SURFACE, lw=1.2, zorder=3)
    # Labels, nudged by hand where points sit on top of each other.
    nudge = {
        "dpnca": (0.07, 0.08), "ncalm": (0.07, 0.0), "wavelm": (0.07, -0.06), "hdclm": (0.07, 0.0),
        "sfnn": (0.07, 0.0), "mashi": (0.07, 0.0), "nen": (0.07, 0.0),
        "fedn": (0.07, 0.05), "olento": (-0.08, 0.0), "harm": (0.07, 0.0), "wave-treatise": (-0.08, 0.0),
        "apeiron": (0.07, 0.0), "drn": (0.07, 0.0), "rglm": (-0.08, 0.0), "chaotic": (0.0, -0.2),
        "pw": (-0.08, 0.0),
    }
    for d, s in scores.items():
        pid = DIR_TO_ID[d]
        dx, dy = nudge[pid]
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        ax.text(float(s["novelty"]) + dx, float(s["performance"]) + dy, SHORT[pid], fontsize=8.5, ha=ha, va="center", color=INK)
    mx = sorted(xs)[len(xs) // 2]; my = sorted(ys)[len(ys) // 2]
    ax.axvline(mx, color=LINE, lw=1, ls="--"); ax.axhline(my, color=LINE, lw=1, ls="--")
    ax.set_xlim(1.5, 6.6); ax.set_ylim(0.5, 6.4)
    ax.set_xlabel("idea strength (novelty axis: mechanism originality, distance from prior art, framing)")
    ax.set_ylabel("evidence strength (performance axis: result quality, scale, baseline fairness)")
    ax.set_title("Idea strength against evidence strength (dashed lines: medians)", loc="left", fontsize=12, pad=12)
    ax.text(6.55, 6.25, "measured and fair", ha="right", va="top", fontsize=9, color=INK3)
    ax.text(1.55, 6.25, "evidence without a new idea", ha="left", va="top", fontsize=9, color=INK3)
    ax.text(6.55, 0.6, "ideas without evidence", ha="right", va="bottom", fontsize=9, color=INK3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "idea_vs_evidence")


def fig_component_share():
    rows = [
        ("Chaotic Reservoir LM — chaos gate", 0.0000268),
        ("DRN — Kuramoto 'resonance'", 0.0016),
        ("DPNCA-LM — all cross-position mixing", 0.083),
        ("olento — opcode-patched weights", 0.127),
        ("Chaotic Reservoir LM — reservoir W_res", 0.202),
        ("DRN — genome (weight generator)", 0.2605),
        ("NCALM v13 — morphogen field", 0.39),
        ("NCALM v7.1 — non-transformer part", 7.7),
        ("APCN-LM — non-transformer part", 30.0),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = range(len(rows))
    colours = [BLUE if "DPNCA" in r[0] else INK3 for r in rows]
    ax.barh(y, [r[1] for r in rows], color=colours, height=0.7)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 100)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows])
    for i, r in enumerate(rows):
        label = f"{r[1]:g}%" if r[1] >= 0.01 else f"{r[1]:.1e}%"
        ax.text(r[1] * 1.25, i, label, va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("share of the model's parameters held by the novel component (log scale)")
    ax.set_title("The novel component owns almost none of the model", loc="left", fontsize=12, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LINE, lw=0.6); ax.set_axisbelow(True)
    save(fig, "component_share")


def fig_timeline(projects):
    rows = sorted(projects, key=lambda p: p["start"])
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    origin = date(2024, 12, 1)
    def days(s):
        y, m, d = map(int, s.split("-")); return (date(y, m, d) - origin).days
    for i, p in enumerate(rows):
        c = THREAD_COLOUR[p["thread"]]
        spans = p.get("spans") or [[p["start"], p["end"]]]
        for s0, s1 in spans:
            a, b = days(s0), days(s1)
            if b - a < 10:
                b = a + 10
            ax.barh(i, b - a, left=a, color=c, height=0.62, alpha=0.95)
        ax.text(b + 6, i, SHORT[p["id"]], va="center", fontsize=8.6, color=INK)
    ax.set_yticks([]); ax.invert_yaxis()
    months = []
    for y in (2024, 2025, 2026):
        for m in range(1, 13):
            d = date(y, m, 1)
            if date(2024, 12, 1) <= d <= date(2026, 11, 1):
                months.append(d)
    ticks = [(d - origin).days for d in months if d.month in (1, 4, 7, 10)]
    labels = [d.strftime("%b %Y") for d in months if d.month in (1, 4, 7, 10)]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-5, (date(2026, 12, 15) - origin).days)
    ax.grid(axis="x", color=LINE, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Eighteen programmes, December 2024 to October 2026", loc="left", fontsize=12, pad=12)
    handles = [Patch(color=THREAD_COLOUR[k], label=lbl) for k, lbl in (
        ("grown", "grown / developmental"), ("physics", "physics-inspired substrates"),
        ("active-inference", "active inference & free energy"), ("local-learning", "local, backprop-free learning"),
        ("current", "after the audit: connectome principles"))]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5)
    save(fig, "timeline")


def fig_loadbearing():
    """Each mechanism as the whole model on one scale with its null arm and the incumbent (filled: the
    default learning rate, two seeds; hollow: each one's best rate, one seed), n-gram floors as rungs;
    and the budget check at three times the bytes."""
    path = ROOT / "experiments" / "loadbearing" / "summary.json"
    if not path.exists():
        return
    S = json.load(open(path))
    exps = [e for e in ["e1_nca", "e2_reservoir", "e3_genome", "e4_kuramoto", "e5_morphogen", "e6_predcode"] if e in S["experiments"]]
    if not exps:
        return
    short = {"e1_nca": "cellular-automaton rule\nDPNCA-LM, NCALM", "e2_reservoir": "fixed random reservoir\nChaotic Reservoir LM", "e3_genome": "genome-made weights\nDRN",
             "e4_kuramoto": "coupled oscillators\nDRN", "e5_morphogen": "morphogen field\nNCALM v13", "e6_predcode": "predictive coding\nAPCN-LM"}
    GREY = "#9ea39f"
    floors = S.get("floors", {}).get("at_budget", {}).get("8000000", {})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 0.78 * len(exps) + 1.9), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.32})
    # ---- the ladder
    for k, name in enumerate(["5-gram", "4-gram", "trigram", "bigram", "unigram"]):
        if name in floors:
            ax.axvline(floors[name], color=INK3, lw=0.9, ls=(0, (3, 4)))
            ax.text(floors[name], -0.72 - 0.26 * (k % 2), name, ha="center", va="bottom", fontsize=8.5, color=INK3,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    for i, e in enumerate(exps):
        ex = S["experiments"][e]; a = ex["arms"]; t = ex.get("tuned", {})
        me, nu, inc = a[ex["mechanism"]]["bpb"], a[ex["null"]]["bpb"], a[ex["incumbent"]]["bpb"]
        ax.plot([min(me, nu), max(me, nu)], [i, i], color=INK3, lw=1.2, alpha=0.6, zorder=1)
        for arm, v, col in ((ex["mechanism"], me, BLUE), (ex["incumbent"], inc, GREY)):
            if arm in t:
                ax.plot([t[arm][1], v], [i, i], color=col, lw=1.2, zorder=2)
                ax.scatter([t[arm][1]], [i], s=52 if col == BLUE else 36, facecolor="white", edgecolor=col, lw=1.4, zorder=4)
        ax.scatter([nu], [i], s=46, color=ORANGE, zorder=3, edgecolor="white", lw=1.2)
        ax.scatter([inc], [i], s=46, color=GREY, zorder=3, edgecolor="white", lw=1.2)
        ax.scatter([me], [i], s=86, color=BLUE, zorder=5, edgecolor="white", lw=1.2)
        bb = dict(facecolor="white", edgecolor="none", pad=1.2)
        ax.text(me, i - 0.22, f"{me:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold", bbox=bb)
        ax.text(nu, i - 0.22, f"{nu:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2, bbox=bb)
        ax.text(inc, i + 0.24, f"{inc:.2f}", ha="center", va="top", fontsize=8.5, color=INK2, bbox=bb)
    ax.set_yticks(range(len(exps))); ax.set_yticklabels([short[e] for e in exps], fontsize=9)
    ax.set_ylim(len(exps) - 0.5, -1.3); ax.set_xlim(1.0, 4.6)
    ax.set_xlabel("bits per byte on the same held-out bytes, lower is better", fontsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Each mechanism as the whole model, its null arm, the incumbent", loc="left", fontsize=11.5, pad=30)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color="w", markerfacecolor=BLUE, markersize=9, label="the mechanism as the model"),
               Line2D([], [], marker="o", color="w", markerfacecolor=ORANGE, markersize=7.5, label="its null arm"),
               Line2D([], [], marker="o", color="w", markerfacecolor=GREY, markersize=7.5, label="the incumbent"),
               Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor=INK3, markersize=7.5, label="hollow: at its best learning rate (one seed)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.2, columnspacing=1.2)
    # ---- the budget check
    lines = []
    for e, arm, col, name in (("e1_nca", "transformer", GREY, "transformer"), ("e1_nca", "nca_shared6", BLUE, "cellular automaton"), ("e5_morphogen", "morphogen", AQUA, "morphogen field")):
        d = S["experiments"].get(e, {}).get("budget_check", {}).get(arm)
        if d and d.get("24m") is not None:
            lines.append((name, col, d["main_budget"], d["24m"]))
    f8, f24 = floors.get("5-gram"), S.get("floors", {}).get("at_budget", {}).get("24000000", {}).get("5-gram")
    if lines:
        if f8 and f24:
            bx.plot([0, 1], [f8, f24], color=INK3, lw=0.9, ls=(0, (3, 4)))
            bx.text(1.04, f24, "5-gram floor", va="center", fontsize=8.5, color=INK3)
        ends = sorted([[v24, name] for name, _, _, v24 in lines])
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.028)
        ypos = {name: y for y, name in ends}
        for name, col, v8, v24 in lines:
            bx.plot([0, 1], [v8, v24], color=col, lw=2.2, zorder=3)
            bx.scatter([0, 1], [v8, v24], s=40, color=col, edgecolor="white", lw=1.2, zorder=4)
            bx.text(-0.05, v8, f"{v8:.2f}", ha="right", va="center", fontsize=8.5, color=INK)
            bx.text(1.05, ypos[name], f"{v24:.2f}  {name}", ha="left", va="center", fontsize=8.5, color=INK)
        bx.set_xlim(-0.35, 2.0); bx.set_xticks([0, 1]); bx.set_xticklabels(["8M bytes", "24M bytes"], fontsize=9)
        lo = min(v for _, _, a_, b_ in lines for v in (a_, b_)); hi = max([v for _, _, a_, b_ in lines for v in (a_, b_)] + [f8 or 0])
        bx.set_ylim(lo - 0.08, hi + 0.08)
        bx.set_yticks([]); bx.tick_params(axis="x", length=0)
        for sp in ("top", "right", "left"):
            bx.spines[sp].set_visible(False)
        bx.set_title("Three times the bytes, each at its best rate:\nthe transformer catches up", loc="left", fontsize=11.5, pad=12)
    else:
        bx.axis("off")
    save(fig, "loadbearing")


if __name__ == "__main__":
    scores, projects = load()
    fig_ranking(scores, projects)
    fig_idea_vs_evidence(scores, projects)
    fig_component_share()
    fig_timeline(projects)
    fig_loadbearing()
    print("wrote", sorted(p.name for p in FIG.iterdir()))
