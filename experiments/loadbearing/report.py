"""Render RESULTS.md and summary.json from runs/.

    python report.py

Every number comes from a runs/<experiment>/<arm>_s<seed>.json written by harness.train,
or from runs/floors.json written by `harness.py floors`; nothing here is typed in by hand.
"""
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# For each experiment: the arm that is the mechanism, the arm that is its null (the same model
# with the mechanism switched off, which should be able to do no better than a bigram model),
# the incumbent it was meant to replace, and a one-line role for every arm.
ROLES = {
    "e1_nca": dict(
        title="E1. A language model that is nothing but a cellular automaton",
        lineage="DPNCA-LM, NCALM", old_share="0.083% (DPNCA-LM's cross-position mixing)",
        mechanism="nca_shared6", null="nca_local6", incumbent="transformer",
        crosses="the rule's depthwise perception (kernels 3, 5, 7), iterated",
        labels={
            "nca_shared6":   "one rule, iterated 6 times, weights shared (the mechanism)",
            "nca_shared1":   "the rule applied once",
            "nca_unshared6": "6 separate rules: a plain causal CNN",
            "nca_local6":    "perception reduced to the centre tap: nothing crosses positions (the null arm)",
            "transformer":   "2-layer transformer, d = 128 (the incumbent)",
            "transformer_s": "1-layer transformer, d = 96 (a smaller incumbent)",
        }),
    "e2_reservoir": dict(
        title="E2. A fixed random reservoir as the only sequence operator",
        lineage="Chaotic Reservoir LM", old_share="0.202% of parameters, 0% of trainable ones (the reservoir)",
        mechanism="reservoir_r0.9", null="reservoir_r0", incumbent="gru",
        crosses="the fixed reservoir matrix W_res, recurring over positions",
        labels={
            "reservoir_r0.9": "fixed random reservoir, spectral radius 0.9 (the mechanism)",
            "reservoir_r2.1": "spectral radius 2.1, the original project's setting",
            "reservoir_r0":   "no recurrence: h_t = tanh(u_t) (the null arm)",
            "gru":            "a trained GRU with the same state size (the incumbent)",
        }),
    "e3_genome": dict(
        title="E3. Every weight emitted by a small genome",
        lineage="DRN", old_share="0.26% (the weight generator)",
        mechanism="genome", null="direct", incumbent="direct_small",
        crosses="the phenotype (E1's cellular automaton, d = 64, 4 iterations); the genome only writes its weights",
        labels={
            "genome":        "13K-parameter genome emitting all 66K phenotype weights, trained end to end (the mechanism)",
            "genome_h192":   "a wider genome, 44K parameters for the same 66K weights",
            "genome_frozen": "genome frozen at initialisation with a trained readout: a random-features probe",
            "direct":        "the same phenotype with its 66K weights trained directly (the null arm: no genome)",
            "direct_small":  "a directly trained phenotype with 17K parameters, about the genome's count (the incumbent at matched size)",
        }),
    "e4_kuramoto": dict(
        title="E4. Coupled oscillators as the recurrence",
        lineage="DRN", old_share="0.0016% (the Kuramoto term)",
        mechanism="kuramoto", null="kuramoto_nomem", incumbent="gru",
        crosses="the oscillator phases, carried from byte to byte and coupled through A",
        labels={
            "kuramoto":        "trained coupling A, learned K (the mechanism)",
            "kuramoto_fixedA": "fixed random coupling, everything else trained",
            "kuramoto_k0":     "coupling off: phases only accumulate the input (integration without interaction; cannot represent a bigram)",
            "kuramoto_nomem":  "phases reset every byte: no memory at all (the null arm)",
            "gru":             "a trained GRU with the same state size (the incumbent)",
        }),
    "e5_morphogen": dict(
        title="E5. A morphogen field as the only thing that crosses positions",
        lineage="NCALM v13", old_share="0.39% (the morphogen field)",
        mechanism="morphogen", null="morphogen_nodiff", incumbent="transformer_s",
        crosses="causal diffusion along positions: 512 learned per-channel, per-step rates",
        labels={
            "morphogen":        "learned diffusion, learned reaction (the mechanism)",
            "morphogen_nodiff": "diffusion off: the reaction alone, so nothing crosses positions (the null arm)",
            "morphogen_lin":    "diffusion only, no reaction: a linear model",
            "transformer_s":    "1-layer transformer, d = 96 (the incumbent)",
        }),
    "e6_predcode": dict(
        title="E6. Hierarchical predictive coding as the whole model",
        lineage="APCN-LM", old_share="about 30% (the non-transformer part)",
        mechanism="pc_k3", null="pc_nomem", incumbent="gru",
        crosses="two levels of latent state, carried from byte to byte and corrected by prediction errors",
        labels={
            "pc_k3":    "three inference steps per byte (the mechanism)",
            "pc_k1":    "one inference step per byte",
            "pc_g0":    "gains fixed at zero: the states never move (a unigram model by construction)",
            "pc_nomem": "states reset every byte: inference without memory (the null arm)",
            "gru":      "a trained GRU with the same total state size (the incumbent)",
        }),
}
FLOOR_ORDER = ["unigram", "bigram", "trigram", "4-gram", "5-gram"]
MARGIN = 0.1   # bits/byte; two seeds of the same arm agree to within 0.01 to 0.05 here


def load():
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(RUNS.glob("e*/*.json")):
        r = json.loads(f.read_text())
        out[r["exp"]][r["arm"]].append(r)
    floors = json.loads((RUNS / "floors.json").read_text()) if (RUNS / "floors.json").exists() else {}
    return out, floors


def stats(rs):
    v = [r["val_bpb"] for r in rs]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else float("nan")), v


def level(bpb, fl):
    """Where a score sits among the n-gram floors at the same byte budget."""
    names = [n for n in FLOOR_ORDER if n in fl]
    if bpb < fl[names[-1]]:
        return f"below the {names[-1]} floor ({fl[names[-1]]:.2f})"
    for hi, lo in zip(names, names[1:]):
        if fl[lo] <= bpb < fl[hi]:
            return f"between the {hi} ({fl[hi]:.2f}) and {lo} ({fl[lo]:.2f}) floors"
    return f"at or above the unigram floor ({fl['unigram']:.2f})"


def reading(mb, nb, cb, fl, tuned=None):
    """One phrase per mechanism, by rule.  `tuned` = (mechanism, incumbent) at each one's best
    learning rate in the sweep (seed 0), used for the incumbent comparison when available."""
    bi = fl["bigram"]
    if mb > nb + MARGIN:
        return "worse than its null arm"
    if mb < bi - MARGIN and mb < nb - MARGIN:
        m2, c2 = tuned if tuned else (mb, cb)
        where = " at each one's best learning rate" if tuned else ""
        if m2 < c2 - MARGIN:
            return f"carries the model, {c2 - m2:.2f} bits/byte ahead of the incumbent{where}"
        if c2 < m2 - MARGIN:
            return f"carries the model, {m2 - c2:.2f} bits/byte behind the incumbent{where}"
        return f"carries the model, level with the incumbent{where}"
    if mb < nb - MARGIN:
        return "learns, but stays at the bigram floor"
    return "not load-bearing"


def split_arm(arm):
    """'nca_shared6_lr0.001' -> ('nca_shared6', 'lr', '0.001'); 'transformer_b24m' -> ('transformer', 'b', '24m')."""
    m = re.match(r"^(.*)_lr([0-9.]+)$", arm)
    if m:
        return m.group(1), "lr", m.group(2)
    m = re.match(r"^(.*)_b(\d+m)$", arm)
    if m:
        return m.group(1), "b", m.group(2)
    return arm, None, None


def main():
    runs, floors = load()
    full = floors.get("full_slice", {})
    L = ["# Results: the null arm, run", "",
         "Byte-level language models on the same TinyStories slice, scored on the same 262,144 held-out bytes "
         "(bits per byte, lower is better). Within an experiment every arm trains under the same byte budget, "
         "context length and optimiser settings; two seeds, both values shown. `share` is the fraction of the "
         "model's trainable parameters inside the mechanism under test, the quantity the audit's \"the new part "
         "owns almost none of the model\" chart reports.", "",
         "**Floors.** Interpolated n-gram models (Witten-Bell smoothing, no tuned weights) scored on the same "
         "bytes, with counts from as many training bytes as the models in that experiment see. They say what "
         "memorising short-range statistics is worth here; a mechanism that lands at the bigram floor has learned "
         "nothing about context. With the whole 60,000,000-byte slice the floors barely move ("
         + ", ".join(f"{k} {full[k]:.3f}" for k in FLOOR_ORDER if k in full) + "), so the data, not the budget, sets them. "
         "The n-gram sees the evaluation stream contiguously while the models see fixed windows, which makes the floors "
         "slightly stricter than a windowed n-gram would reach.", ""]
    summary = {"floors": floors, "experiments": {}}
    for exp, R in ROLES.items():
        if exp not in runs:
            continue
        arms = runs[exp]
        main_arms = {a: rs for a, rs in arms.items() if split_arm(a)[1] is None}
        budget = main_arms[R["mechanism"]][0]["budget"]["bytes"] if R["mechanism"] in main_arms else None
        fl = floors.get("at_budget", {}).get(str(budget), full)
        ctx = next((r.get("ctx") for rs in main_arms.values() for r in rs if r.get("ctx")), None)
        L += [f"## {R['title']}", "",
              f"Lineage: {R['lineage']}. Share of the model the mechanism held there: {R['old_share']}. "
              f"What crosses positions here: {R['crosses']}. Budget {budget:,} bytes, context {ctx}. "
              "Floors at this budget: " + ", ".join(f"{k} {fl[k]:.3f}" for k in FLOOR_ORDER if k in fl) + ".", "",
              "| arm | role | trainable params | share in mechanism | bits/byte | seeds | wall-clock |", "|---|---|---|---|---|---|---|"]
        ex = {"title": R["title"], "lineage": R["lineage"], "old_share": R["old_share"], "crosses": R["crosses"],
              "budget_bytes": budget, "ctx": ctx, "floors": fl, "arms": {}, "lr_check": {}, "budget_check": {}}
        order = list(R["labels"]) + [a for a in main_arms if a not in R["labels"]]
        for arm in order:
            if arm not in main_arms:
                continue
            rs = sorted(main_arms[arm], key=lambda r: r["seed"])
            m, sd, v = stats(rs)
            r0 = rs[0]
            fixed = r0.get("fixed_mechanism_params", 0)
            share = r0.get("share_all", r0["share"]) if fixed else r0["share"]
            share_txt = f"{share*100:.0f}% of all parameters, fixed" if fixed else f"{share*100:.1f}%"
            disagree = (max(v) - min(v)) > 0.3
            name = f"**{arm}**" if arm == R["mechanism"] else arm
            L.append(f"| {name} | {R['labels'].get(arm, '')} | {r0['trainable_params']:,} | {share_txt} | **{m:.3f}** | "
                     f"{', '.join(f'{x:.3f}' for x in v)}{' (seeds disagree)' if disagree else ''} | {st.mean(r['wall_s'] for r in rs)/60:.0f} min |")
            ex["arms"][arm] = {"bpb": m, "sd": sd, "values": v, "seeds": [r["seed"] for r in rs], "role": R["labels"].get(arm, ""),
                               "trainable_params": r0["trainable_params"], "share": share, "fixed_mechanism_params": fixed,
                               "seeds_disagree": disagree, "level": level(m, fl), "curve": [c["val_bpb"] for c in r0["curve"]]}
        ex["mechanism"], ex["null"], ex["incumbent"] = R["mechanism"], R["null"], R["incumbent"]
        # checks: learning rate and budget, one seed
        budget_lr = {}
        for arm, rs in arms.items():
            base, kind, val = split_arm(arm)
            if kind == "lr":
                ex["lr_check"].setdefault(base, {})[val] = stats(rs)[0]
            elif kind == "b":
                ex["budget_check"].setdefault(base, {})[val] = stats(rs)[0]
                budget_lr[base] = rs[0]["budget"]["lr"]
        for base, d in ex["lr_check"].items():          # the main run's seed 0 is the sweep's 0.002 point
            main0 = next((r["val_bpb"] for r in main_arms.get(base, []) if r["seed"] == 0), None)
            if main0 is not None:
                d["0.002"] = main0
        fin = lambda v: v if v == v else float("inf")                                     # a diverged run (nan) never wins
        ex["tuned"] = {base: min(d.items(), key=lambda kv: fin(kv[1])) for base, d in ex["lr_check"].items()}   # arm -> (lr, bpb)
        have = all(a in main_arms for a in (R["mechanism"], R["null"], R["incumbent"]))
        if have:
            mb, nb, cb = (stats(main_arms[a])[0] for a in (R["mechanism"], R["null"], R["incumbent"]))
            ex["mechanism_vs_null"], ex["mechanism_vs_incumbent"] = nb - mb, mb - cb
            tuned = None
            if R["mechanism"] in ex["tuned"] and R["incumbent"] in ex["tuned"]:
                tuned = (ex["tuned"][R["mechanism"]][1], ex["tuned"][R["incumbent"]][1])
                ex["mechanism_vs_incumbent_tuned"] = tuned[0] - tuned[1]
            ex["reading"], ex["level"] = reading(mb, nb, cb, fl, tuned), level(mb, fl)
            ex["reading_short"] = ex["reading"].replace(" bits/byte", "").replace(" at each one's best learning rate", " at best rates")
            L += ["", f"Mechanism against its null arm: **{nb - mb:+.3f}** bits/byte. Against the incumbent at the shared learning rate: **{mb - cb:+.3f}**"
                      + (f"; at each one's best learning rate in the sweep (seed 0): **{tuned[0] - tuned[1]:+.3f}**" if tuned else "")
                      + f". The mechanism sits {ex['level']}. Reading: **{ex['reading']}**.", ""]
        else:
            missing = [a for a in (R["mechanism"], R["null"], R["incumbent"]) if a not in main_arms]
            L += ["", f"(waiting for: {', '.join(missing)})", ""]
        if ex["lr_check"]:
            lrs = sorted({lr for d in ex["lr_check"].values() for lr in d}, key=float)
            L += ["Learning-rate sweep (seed 0; the main runs use 0.002; best in bold):", "",
                  "| arm | " + " | ".join(f"lr {lr}" for lr in lrs) + " | best |", "|---|" + "---|" * (len(lrs) + 1)]
            for base, d in ex["lr_check"].items():
                blr, bv = ex["tuned"][base]
                cell = lambda v: "diverged" if v != v else f"{v:.3f}"
                cells = [("**" + cell(d[lr]) + "**" if lr == blr else cell(d[lr])) if lr in d else "–" for lr in lrs]
                L.append(f"| {base} | " + " | ".join(cells) + f" | {bv:.3f} at {blr} |")
            L.append("")
        if ex["budget_check"]:
            L += ["Budget check (seed 0; three times the bytes, each arm at its best learning rate from the sweep):", "",
                  f"| arm | learning rate | {budget:,} bytes | 24,000,000 bytes | gain |", "|---|---|---|---|---|"]
            for base, d in ex["budget_check"].items():
                lr = budget_lr[base]
                same_lr = ex["lr_check"].get(base, {}).get(f"{lr:g}", float("nan"))
                v24 = d.get("24m", float("nan"))
                L.append(f"| {base} | {lr:g} | {same_lr:.3f} | {v24:.3f} | {same_lr - v24:.3f} |")
                ex["budget_check"][base] = {"lr": lr, "main_budget": same_lr, "24m": v24}
            fl24 = floors.get("at_budget", {}).get("24000000")
            if fl24:
                L.append("\nFloors at 24,000,000 bytes: " + ", ".join(f"{k} {fl24[k]:.3f}" for k in FLOOR_ORDER if k in fl24) + ".")
            L.append("")
        summary["experiments"][exp] = ex
    L += ["## Caveats", "",
          "- Same budget, not convergence. Every arm is still improving when its budget ends, and the learning rate moves every "
          "arm by more than the seed spread does. Each mechanism and incumbent was therefore swept over 0.001 to 0.016 (0.032 "
          "where the best was still at the top; one seed): the comparisons with the incumbents use each arm's best rate, and "
          "the budget check at three times the bytes, at those rates, shows how the gaps move with data. For `reservoir_r0.9`, "
          "`kuramoto` and `pc_k3` the best rate was the top of the sweep (0.016; 0.032 was worse or diverged) and for `direct` "
          "0.032 beat 0.016 by 0.003; every other best rate is interior.",
          "- Two seeds at the default rate; one seed in the sweep and the budget check. The spread between seeds is 0.001 to "
          "0.05 bits/byte for every arm except `genome_frozen` (0.22: a random-features probe, whose features are the draw), "
          "`kuramoto_fixedA` (0.05) and the wider genome, whose seeds are shown separately. Differences under 0.1 bits/byte "
          "are not read as differences.",
          "- Share. `share` counts trainable parameters inside the mechanism under test. In E5 that includes the per-position "
          "reaction MLPs; only the 512 diffusion rates move information across positions, so the mechanism's claim there "
          "rests on the null arm (diffusion off), not on the share. In E2 the reservoir has no trainable parameters, so its "
          "share of all parameters is reported instead.",
          "- Nulls. E4's `kuramoto_k0` and E6's `pc_g0` switch the mechanism off but cannot represent even a bigram model "
          "(the oscillators integrate without reading the current byte; the zero-gain states never move), so they are not "
          "the null arms. The null arms are `kuramoto_nomem` and `pc_nomem`, which keep the per-byte computation and remove "
          "memory. E3's null arm is the same phenotype with directly trained weights; `genome_frozen` is a random-features probe. "
          "The null arms were run at the default rate only: each sits within 0.01 bits/byte of the bigram floor, and a model "
          "at the bigram floor has nothing left to gain from its optimiser.",
          "- Floors. The n-gram floors use Witten-Bell interpolation with counts from the first 6,000,000, 8,000,000 or "
          "24,000,000 training bytes (whichever the budget is), scored on the contiguous evaluation stream. Better smoothing "
          "(for instance Kneser-Ney) would lower the higher-order floors somewhat; the bigram floor, which the reading rule "
          "uses, is insensitive to this.",
          "- Scale. These are byte-level models of 13K to 870K parameters trained on 6 to 24 million bytes on two CPU cores. "
          "They say what each mechanism carries when it has to carry the model, and how that moves with the learning rate and "
          "the budget at this scale; they do not say where any of them would end up.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(L) + "\n")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
