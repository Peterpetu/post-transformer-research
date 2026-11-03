"""E4: a language model whose hidden state is a population of coupled oscillators.

DRN carried a Kuramoto "resonance" term that was detached from the graph and converted
to a Python scalar, so its gradient was exactly zero and it held 0.0016% of the
parameters.  Here the Kuramoto coupling is the recurrence:

    theta_{t+1} = theta_t + omega(x_t) + (K / N) * sum_j A_ij sin(theta_j - theta_i)
    logits_t    = readout([cos theta_t, sin theta_t])

The coupling matrix A is trained, and it is the only interaction between oscillators,
so it is the mechanism.  The natural frequencies omega(x) are an embedding of the
current byte.  With K = 0 the oscillators are independent and each phase is simply the
running sum of the frequencies it has been driven with: a bag of accumulated inputs,
which is the null arm.

Arms (two seeds):
    kuramoto         trained coupling A, learned K                   (the mechanism)
    kuramoto_fixedA  random fixed coupling, everything else trained  (is the learned coupling what matters?)
    kuramoto_k0      coupling off; phases still accumulate the input (integration without interaction)
    kuramoto_nomem   phases reset every byte: no memory at all        (the bigram-level null arm)
    gru              a trained GRU with the same state size          (the incumbent)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import harness
from harness import VOCAB, Budget, Bytes, TinyGRU, n_params, parse_common, train


class KuramotoLM(nn.Module):
    def __init__(self, n_osc=256, coupling="trained", k_init=1.0):
        super().__init__()
        self.n = n_osc
        self.omega = nn.Embedding(VOCAB, n_osc)
        nn.init.normal_(self.omega.weight, std=0.5)
        A = torch.randn(n_osc, n_osc) / math.sqrt(n_osc)
        if coupling == "trained":
            self.A = nn.Parameter(A)
        else:
            self.register_buffer("A", A)
        self.coupling = coupling
        self.logK = nn.Parameter(torch.tensor(math.log(k_init))) if coupling not in ("off", "nomem") else None
        self.readout = nn.Sequential(nn.Linear(2 * n_osc, 256), nn.GELU(), nn.Linear(256, VOCAB))

    def forward(self, x):
        w = self.omega(x)                       # (B, T, N) natural frequencies per token
        B, T, N = w.shape
        theta = torch.zeros(B, N)
        outs = []
        for t in range(T):
            if self.coupling == "nomem":
                theta = w[:, t]                                   # no accumulation, no coupling: a bigram model in oscillator clothing
            elif self.coupling != "off":
                s, c = torch.sin(theta), torch.cos(theta)
                # sum_j A_ij sin(theta_j - theta_i) = cos(theta_i) * (A sin theta)_i - sin(theta_i) * (A cos theta)_i
                coup = c * (s @ self.A.T) - s * (c @ self.A.T)
                theta = theta + w[:, t] + torch.exp(self.logK) / N * coup
            else:
                theta = theta + w[:, t]
            theta = torch.remainder(theta + math.pi, 2 * math.pi) - math.pi
            outs.append(torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1))
        return self.readout(torch.stack(outs, dim=1))


ARMS = {
    "kuramoto":        lambda d: KuramotoLM(256, coupling="trained"),
    "kuramoto_fixedA": lambda d: KuramotoLM(256, coupling="fixed"),
    "kuramoto_k0":     lambda d: KuramotoLM(256, coupling="off"),
    "kuramoto_nomem":  lambda d: KuramotoLM(256, coupling="nomem"),   # phases reset every byte: no memory at all, the bigram-level null
    "gru":             lambda d: TinyGRU(d, hidden=256),
}
IS_MECH = lambda name: name in ("A", "logK")

if __name__ == "__main__":
    a = parse_common(["--ctx", "64", "--batch", "64"] + __import__("sys").argv[1:])
    harness.set_ctx(a.ctx)
    data = Bytes()
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in arms:
            if (harness.RUNS / "e4_kuramoto" / f"{arm}{a.suffix}_s{seed}.json").exists():
                print(f"{arm} s{seed}: done, skipping", flush=True); continue
            torch.manual_seed(seed)
            m = ARMS[arm](a.d)
            print(f"{arm}: {n_params(m):,} trainable params", flush=True)
            train(m, data, Budget(bytes=a.budget, batch=a.batch, lr=a.lr), seed, arm + a.suffix, "e4_kuramoto",
                  IS_MECH, extra={"params": n_params(m)})
