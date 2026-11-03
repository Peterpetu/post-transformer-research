"""E1: a language model that is nothing but a neural cellular automaton.

In DPNCA-LM the cellular-automaton rule owned 0.083% of the parameters: the only
cross-position operator was a bank of depthwise convolutions, and everything else
(feed-forward blocks, embeddings) was a transformer skeleton.  Here the rule is the
model.  One rule, shared across every iteration, carries all of the computation
between the embedding and the readout:

    perceive: causal depthwise convolutions over positions (kernels 3, 5, 7) -> 3d channels
    mix:      Linear(3d, 2d) -> GELU -> Linear(2d, d)      (the rule's parameters)
    update:   h <- LayerNorm(h + step * mix(perceive(h)))

Arms (all at the same byte budget; two seeds):
    nca_shared6     the rule, iterated 6 times, weights shared        (the mechanism)
    nca_shared1     the rule applied once                               (is iteration load-bearing?)
    nca_unshared6   6 separate rules, i.e. a plain causal CNN           (does sharing cost?)
    nca_local6      perception replaced by the centre tap only          (the null arm: no cross-position information)
    transformer_s   a 1-layer transformer of about the same size        (the incumbent, small)
    transformer     a 2-layer transformer, larger                       (the incumbent)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import harness
from harness import VOCAB, Budget, Bytes, TinyTransformer, n_params, parse_common, train


class Perceive(nn.Module):
    """Causal depthwise convolutions with kernels 3, 5, 7 -> concatenated (3d channels)."""
    def __init__(self, d, kernels=(3, 5, 7), local_only=False):
        super().__init__()
        self.local_only = local_only
        self.kernels = kernels
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, groups=d, bias=True) for k in kernels])

    def forward(self, h):               # h: (B, T, d)
        if self.local_only:
            return torch.cat([h, h, h], dim=-1)
        x = h.transpose(1, 2)           # (B, d, T)
        outs = [c(F.pad(x, (k - 1, 0))) for c, k in zip(self.convs, self.kernels)]  # left-pad: causal
        return torch.cat(outs, dim=1).transpose(1, 2)


class Rule(nn.Module):
    def __init__(self, d, local_only=False):
        super().__init__()
        self.perceive = Perceive(d, local_only=local_only)
        self.mix = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, step):
        return self.norm(h + step * self.mix(self.perceive(h)))


class NCAOnlyLM(nn.Module):
    def __init__(self, d=128, steps=6, shared=True, local_only=False):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.steps = steps
        n_rules = 1 if shared else steps
        self.rule = nn.ModuleList([Rule(d, local_only) for _ in range(n_rules)])
        self.step_size = nn.Parameter(torch.full((steps,), 0.5))
        self.out = nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.emb(x)
        for s in range(self.steps):
            r = self.rule[0] if len(self.rule) == 1 else self.rule[s]
            h = r(h, self.step_size[s])
        return self.out(h)


ARMS = {
    "nca_shared6":   lambda d: NCAOnlyLM(d, steps=6, shared=True),
    "nca_shared1":   lambda d: NCAOnlyLM(d, steps=1, shared=True),
    "nca_unshared6": lambda d: NCAOnlyLM(d, steps=6, shared=False),
    "nca_local6":    lambda d: NCAOnlyLM(d, steps=6, shared=True, local_only=True),
    "transformer_s": lambda d: TinyTransformer(d=96, layers=1, heads=4),
    "transformer":   lambda d: TinyTransformer(d=128, layers=2, heads=4),
}
IS_MECH = lambda name: name.startswith("rule.") or name.startswith("step_size")

if __name__ == "__main__":
    a = parse_common()
    harness.set_ctx(a.ctx)
    data = Bytes()
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in arms:
            if (harness.RUNS / "e1_nca" / f"{arm}{a.suffix}_s{seed}.json").exists():
                print(f"{arm} s{seed}: done, skipping", flush=True); continue
            torch.manual_seed(seed)
            m = ARMS[arm](a.d)
            print(f"{arm}: {n_params(m):,} params", flush=True)
            train(m, data, Budget(bytes=a.budget, batch=a.batch, lr=a.lr), seed, arm + a.suffix, "e1_nca",
                  IS_MECH if arm.startswith("nca") else (lambda n: False), extra={"d": a.d, "params": n_params(m)})
