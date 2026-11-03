"""E5: a language model in which a morphogen field is the only thing that crosses positions.

NCALM's v13 added a "morphogen field" to a gated transformer; it held 0.39% of the
parameters and the transformer did the work.  Here the field is the model's whole
memory.  Tokens enter a field m (C channels per position) that evolves by an iterated
reaction-diffusion rule whose diffusion is causal over positions:

    diffuse:  m_t <- (1 - lambda) x_t + lambda m_{t-1}      (an exponential moving average over earlier positions; lambda per channel, learned)
    react:    m   <- m + MLP(m)                               (per position)

repeated S times, and the byte is predicted from the field at its position.  Diffusion
is the only operator that moves information between positions; the reaction is local.
The null arm fixes lambda = 0, so nothing diffuses and the model can only see the
current byte.

Arms (two seeds):
    morphogen        learned diffusion rates, learned reaction            (the mechanism)
    morphogen_nodiff lambda fixed at 0                                    (the null arm)
    morphogen_lin    diffusion only, no reaction MLP                      (is the nonlinearity load-bearing?)
    transformer_s    the 1-layer transformer control from E1              (the incumbent)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import harness
from harness import VOCAB, Budget, Bytes, TinyTransformer, n_params, parse_common, train


class MorphogenLM(nn.Module):
    def __init__(self, C=192, steps=6, diffuse=True, react=True):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, C)
        self.steps, self.diffuse_on, self.react_on = steps, diffuse, react
        # one rate per channel and per step; sigmoid keeps it in (0, 1)
        self.rate_logit = nn.Parameter(torch.randn(steps, C) * 0.5) if diffuse else None
        self.react = nn.ModuleList([nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C)) for _ in range(steps)]) if react else None
        self.norm = nn.LayerNorm(C)
        self.out = nn.Linear(C, VOCAB)

    def forward(self, x):
        m = self.emb(x)                                   # (B, T, C)
        for s in range(self.steps):
            if self.diffuse_on:
                lam = torch.sigmoid(self.rate_logit[s])   # (C,)
                # causal exponential diffusion along positions, m_t <- (1-lam) m_t + lam m_{t-1},
                # written out as the equivalent causal convolution with the kernel (1-lam) lam^k
                T = m.shape[1]
                k = torch.arange(T, dtype=m.dtype)
                kernel = (1 - lam).unsqueeze(1) * lam.unsqueeze(1) ** k.unsqueeze(0)      # (C, T): (1-lam) lam^k
                X = torch.fft.rfft(m.transpose(1, 2), n=2 * T)                             # zero-padded, so the circular product is the linear (causal) convolution
                K = torch.fft.rfft(kernel, n=2 * T)
                m = torch.fft.irfft(X * K, n=2 * T)[..., :T].transpose(1, 2)
            if self.react_on:
                m = m + self.react[s](m)
        return self.out(self.norm(m))


ARMS = {
    "morphogen":        lambda d: MorphogenLM(128, 4, diffuse=True, react=True),
    "morphogen_nodiff": lambda d: MorphogenLM(128, 4, diffuse=False, react=True),
    "morphogen_lin":    lambda d: MorphogenLM(128, 4, diffuse=True, react=False),
    "transformer_s":    lambda d: TinyTransformer(d=96, layers=1, heads=4),
}
IS_MECH = lambda name: name.startswith("rate_logit") or name.startswith("react.")

if __name__ == "__main__":
    a = parse_common()
    harness.set_ctx(a.ctx)
    data = Bytes()
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in arms:
            if (harness.RUNS / "e5_morphogen" / f"{arm}{a.suffix}_s{seed}.json").exists():
                print(f"{arm} s{seed}: done, skipping", flush=True); continue
            torch.manual_seed(seed)
            m = ARMS[arm](a.d)
            print(f"{arm}: {n_params(m):,} trainable params", flush=True)
            train(m, data, Budget(bytes=a.budget, batch=a.batch, lr=a.lr), seed, arm + a.suffix, "e5_morphogen",
                  IS_MECH if arm.startswith("morphogen") else (lambda n: False), extra={"params": n_params(m)})
