"""E2: a language model whose only sequence operator is a fixed random reservoir.

The Chaotic Reservoir LM kept a frozen sparse "chaotic" reservoir between a trainable
input projection and a trainable output head, and the audit found that its sequence
axis had been flattened into the batch, so the reservoir never recurred over time and
the trained model was a rank-512 bigram table.  Here the reservoir recurs:

    u_t = W_in emb(x_t)                            (trainable)
    h_t = tanh(W_res h_{t-1} + u_t)                (W_res fixed random, spectral radius rho)
    logits_t = readout(h_t)                        (trainable)

The reservoir is the only thing that carries information across positions.  Its
parameters are not trained, so the share of *trainable* parameters it holds is zero by
construction; the record also reports its share of all parameters.

Arms (two seeds):
    reservoir_r0.9   rho = 0.9, the usual edge-of-chaos setting   (the mechanism)
    reservoir_r2.1   rho = 2.1, the original project's setting     (deep in the chaotic regime)
    reservoir_r0     rho = 0: no recurrence, h_t = tanh(u_t)       (the null arm; a bigram model with extra steps)
    gru              a trained GRU with the same state size        (trained recurrence, the incumbent)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import harness
from harness import VOCAB, Budget, Bytes, TinyGRU, n_params, parse_common, train


class ReservoirLM(nn.Module):
    def __init__(self, d=128, hidden=256, rho=0.9, density=0.1, seed=0):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.w_in = nn.Linear(d, hidden, bias=False)
        self.readout = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, VOCAB))
        g = torch.Generator().manual_seed(1000 + seed)
        W = torch.randn(hidden, hidden, generator=g) * (torch.rand(hidden, hidden, generator=g) < density)
        if rho > 0:
            eig = torch.linalg.eigvals(W).abs().max()
            W = W * (rho / eig)
        else:
            W = torch.zeros_like(W)
        self.register_buffer("w_res", W)   # fixed: the mechanism
        self.rho = rho

    def forward(self, x):
        u = self.w_in(self.emb(x))                       # (B, T, H)
        B, T, H = u.shape
        h = torch.zeros(B, H)
        hs = []
        for t in range(T):
            h = torch.tanh(h @ self.w_res.T + u[:, t])
            hs.append(h)
        return self.readout(torch.stack(hs, dim=1))


ARMS = {
    "reservoir_r0.9": lambda d, s: ReservoirLM(d, rho=0.9, seed=s),
    "reservoir_r2.1": lambda d, s: ReservoirLM(d, rho=2.1, seed=s),
    "reservoir_r0":   lambda d, s: ReservoirLM(d, rho=0.0, seed=s),
    "gru":            lambda d, s: TinyGRU(d, hidden=256),
}

if __name__ == "__main__":
    a = parse_common(["--ctx", "64", "--batch", "64"] + __import__("sys").argv[1:])
    harness.set_ctx(a.ctx)
    data = Bytes()
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in arms:
            if (harness.RUNS / "e2_reservoir" / f"{arm}{a.suffix}_s{seed}.json").exists():
                print(f"{arm} s{seed}: done, skipping", flush=True); continue
            torch.manual_seed(seed)
            m = ARMS[arm](a.d, seed)
            fixed = int(m.w_res.numel()) if hasattr(m, "w_res") else 0
            print(f"{arm}: {n_params(m):,} trainable params, {fixed:,} fixed in the reservoir", flush=True)
            train(m, data, Budget(bytes=a.budget, batch=a.batch, lr=a.lr), seed, arm + a.suffix, "e2_reservoir",
                  lambda n: False, extra={"d": a.d, "params": n_params(m), "fixed_mechanism_params": fixed,
                                          "share_all": fixed / (fixed + n_params(m))})
