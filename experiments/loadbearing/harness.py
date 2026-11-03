"""Shared harness for the load-bearing experiments.

Every experiment in this directory trains byte-level language models on the same
TinyStories slice under the same budget, evaluates them on the same held-out bytes,
and records, for every model, the share of its trainable parameters that sit inside
the mechanism under test.  That share is the quantity the write-up's "the new part
owns almost none of the model" chart reports; these experiments turn it around.

    python harness.py floors            # n-gram floors on the evaluation bytes
    python <experiment>.py --budget ... # see each experiment's docstring

Data: data/train.txt (the first 60,000,000 bytes of the TinyStories training text),
data/val.txt (1,597,917 bytes of its validation text).  Evaluation uses the first
EVAL_BYTES of val.txt in fixed, non-overlapping windows, so every arm of every
experiment is scored on exactly the same bytes.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = HERE / "runs"
CTX = 256            # context length in bytes (default; recurrent experiments use 64, see set_ctx)
EVAL_BYTES = 262_144 # 1,024 windows of 256 bytes
VOCAB = 256

torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.set_flush_denormal(True)   # a null arm whose state sits at ~1e-13 would otherwise crawl through subnormal arithmetic


def set_ctx(n: int) -> None:
    """Context length for this experiment.  Every arm of an experiment trains and is
    evaluated at the same length, on the same held-out bytes; the recurrent families
    (E2, E4, E6) use 64 so that their sequential loops fit the CPU budget."""
    global CTX
    CTX = n


# ----------------------------------------------------------------------------- data
class Bytes:
    def __init__(self):
        self.train = np.frombuffer((DATA / "train.txt").read_bytes(), dtype=np.uint8)
        val = np.frombuffer((DATA / "val.txt").read_bytes(), dtype=np.uint8)
        n = EVAL_BYTES + 1
        self.val = val[:n]

    def batch(self, rng: np.random.Generator, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        CTX = globals()["CTX"]
        idx = rng.integers(0, len(self.train) - CTX - 1, size=B)
        x = np.stack([self.train[i:i + CTX] for i in idx]).astype(np.int64)
        y = np.stack([self.train[i + 1:i + CTX + 1] for i in idx]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def eval_windows(self) -> tuple[torch.Tensor, torch.Tensor]:
        CTX = globals()["CTX"]
        n = (len(self.val) - 1) // CTX
        x = self.val[:n * CTX].reshape(n, CTX).astype(np.int64)
        y = self.val[1:n * CTX + 1].reshape(n, CTX).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def ngram_floors(data: Bytes, train_bytes: int | None = None) -> dict:
    """Interpolated n-gram (n = 1..5) bits per byte on exactly the bytes the models are
    scored on (the contiguous evaluation stream), with counts from the first
    `train_bytes` bytes of the training slice (all of it when None) and Witten-Bell
    interpolation, which has no tunable weights:

        P_n(w | c) = (count(c, w) + T(c) * P_{n-1}(w | c')) / (N(c) + T(c))

    N(c) is the number of times context c was seen, T(c) the number of distinct bytes
    that followed it; an unseen context falls back to the lower order.  The n-gram sees
    the stream contiguously while the models see fixed windows, so these floors are
    slightly stricter (lower) than a windowed n-gram would reach."""
    tr = data.train[:train_bytes].astype(np.int64) if train_bytes else data.train.astype(np.int64)
    CTX = globals()["CTX"]
    n = ((len(data.val) - 1) // CTX) * CTX
    ev = data.val[:n + 1].astype(np.int64)        # contiguous: targets ev[1:] are the n scored bytes

    def pack(a, k):
        key = np.zeros(len(a) - k + 1, dtype=np.int64)
        for j in range(k):
            key = key * 256 + a[j:len(a) - k + 1 + j]
        return key

    def lookup(keys, vals, q):
        i = np.searchsorted(keys, q)
        i[i >= len(keys)] = 0
        return np.where(keys[i] == q, vals[i], 0).astype(np.float64)

    uni = np.bincount(tr, minlength=256).astype(np.float64)
    p1 = (uni + 0.5) / (uni.sum() + 128)
    p_prev = p1[ev[1:]]                            # probability of every target under the previous order
    out = {"unigram": float(-np.log2(p_prev).mean())}
    names = {2: "bigram", 3: "trigram", 4: "4-gram", 5: "5-gram"}
    for k in range(2, 6):
        kg = pack(tr, k)
        keys, cnt = np.unique(kg, return_counts=True)            # count(c, w)
        ckeys, ccnt = np.unique(kg // 256, return_counts=True)   # N(c)
        tkeys, tcnt = np.unique(keys // 256, return_counts=True) # T(c)
        ekg = pack(ev, k)                                         # k-grams ending at targets ev[k-1:]
        c = lookup(keys, cnt, ekg); N = lookup(ckeys, ccnt, ekg // 256); T = lookup(tkeys, tcnt, ekg // 256)
        pp = p_prev[k - 2:]
        pk = np.where(N > 0, (c + T * pp) / np.maximum(N + T, 1.0), pp)
        full = p_prev.copy(); full[k - 2:] = pk                   # the first k-2 targets keep the lower order
        out[names[k]] = float(-np.log2(full).mean())
        p_prev = full
    out["train_bytes"] = int(len(tr))
    return out


# ----------------------------------------------------------------------------- training
@dataclass
class Budget:
    bytes: int = 6_000_000     # training bytes seen (B * CTX * steps)
    batch: int = 16
    lr: float = 2e-3
    warmup: int = 100
    clip: float = 1.0
    weight_decay: float = 0.05
    eval_every: int = 0        # 0 = only at the end (plus a few fixed checkpoints)

    @property
    def steps(self) -> int:
        return self.bytes // (self.batch * globals()["CTX"])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def mechanism_share(m: nn.Module, is_mechanism) -> dict:
    """Fraction of trainable parameters inside the mechanism under test.
    `is_mechanism(name) -> bool` decides by parameter name."""
    mech = tot = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        tot += p.numel()
        if is_mechanism(name):
            mech += p.numel()
    return {"mechanism_params": mech, "trainable_params": tot, "share": mech / max(1, tot)}


@torch.no_grad()
def evaluate(model: nn.Module, data: Bytes, bs: int = 64) -> float:
    model.eval()
    x, y = data.eval_windows()
    tot = 0.0
    for i in range(0, len(x), bs):
        logits = model(x[i:i + bs])
        tot += F.cross_entropy(logits.reshape(-1, VOCAB), y[i:i + bs].reshape(-1), reduction="sum").item()
    model.train()
    return tot / y.numel() / math.log(2)


def train(model: nn.Module, data: Bytes, budget: Budget, seed: int, tag: str, exp: str,
          is_mechanism, extra: dict | None = None) -> dict:
    """Train under the budget, evaluate on the fixed held-out windows, write runs/<exp>/<tag>_s<seed>.json."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    share = mechanism_share(model, is_mechanism)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=budget.lr,
                            betas=(0.9, 0.95), weight_decay=budget.weight_decay)
    steps = budget.steps
    def lr_at(s):
        if s < budget.warmup:
            return budget.lr * (s + 1) / budget.warmup
        k = (s - budget.warmup) / max(1, steps - budget.warmup)
        return budget.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * k)))
    checkpoints = sorted(set([steps // 4, steps // 2, 3 * steps // 4, steps]))
    curve, t0, losses = [], time.time(), []
    for s in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = data.batch(rng, budget.batch)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if budget.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), budget.clip)
        opt.step()
        losses.append(loss.item())
        if s in checkpoints or (budget.eval_every and s % budget.eval_every == 0):
            bpb = evaluate(model, data)
            curve.append({"step": s, "bytes": s * budget.batch * CTX, "train_bits": float(np.mean(losses[-50:]) / math.log(2)), "val_bpb": bpb, "sec": time.time() - t0})
            print(f"[{exp}/{tag} s{seed}] step {s}/{steps}  train {curve[-1]['train_bits']:.3f}  val {bpb:.3f} bpb  {time.time()-t0:.0f}s", flush=True)
    rec = {"exp": exp, "arm": tag, "seed": seed, "ctx": globals()["CTX"], "budget": asdict(budget), **share,
           "val_bpb": curve[-1]["val_bpb"], "curve": curve, "wall_s": time.time() - t0,
           "torch": torch.__version__, "threads": torch.get_num_threads(), **(extra or {})}
    out = RUNS / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    return rec


# ----------------------------------------------------------------------------- controls
class TinyTransformer(nn.Module):
    """The control everything is measured against: a small pre-norm causal transformer."""
    def __init__(self, d=128, layers=2, heads=4, ff=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d), "attn": nn.MultiheadAttention(d, heads, batch_first=True),
                "ln2": nn.LayerNorm(d), "ff": nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))}))
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, VOCAB)
        self.register_buffer("mask", torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T))
        for b in self.blocks:
            a = b["ln1"](h)
            a, _ = b["attn"](a, a, a, attn_mask=self.mask[:T, :T], need_weights=False)
            h = h + a
            h = h + b["ff"](b["ln2"](h))
        return self.out(self.ln(h))


class TinyGRU(nn.Module):
    """A trained recurrent control with the same state size as a recurrent mechanism."""
    def __init__(self, d=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.out = nn.Linear(hidden, VOCAB)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h)


def parse_common(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=6_000_000, help="training bytes per run")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--arms", type=str, default="all")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--suffix", type=str, default="", help="appended to the arm name in the record (for learning-rate checks)")
    ap.add_argument("--d", type=int, default=128)
    return ap.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "floors":
        # floors at the byte budgets the models train under (counts from as many training
        # bytes as a model sees) and on the whole 60,000,000-byte slice
        d = Bytes()
        fl = {"eval_bytes": EVAL_BYTES, "smoothing": "Witten-Bell interpolation",
              "at_budget": {str(b): ngram_floors(d, b) for b in (6_000_000, 8_000_000, 24_000_000)},
              "full_slice": ngram_floors(d)}
        RUNS.mkdir(exist_ok=True)
        (RUNS / "floors.json").write_text(json.dumps(fl, indent=1))
        print(json.dumps(fl, indent=1))
