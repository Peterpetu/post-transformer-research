"""E6: a language model that is nothing but hierarchical predictive coding.

APCN-LM fused predictive-coding cells (diagonal Kalman filters) with standard transformer
blocks; the transformer was about 72% of the model, the Kalman gain pinned itself at
its floor, and the cells collapsed to a near-no-op.  Here the cells are the model.  Two
levels of latent state are carried across positions and corrected by prediction errors:

    level 1 predicts the input embedding:  p1 = W1 z1,   e1 = emb(x_t) - p1
    level 2 predicts level 1:              p2 = W2 z2,   e2 = z1 - p2
    inference (K steps per byte, states start from the previous byte's):
        z1 <- z1 + g1 * (W1^T e1 - e2)
        z2 <- z2 + g2 * (W2^T e2)
    logits_t = readout([z1, z2])

g1 and g2 are learned per-channel gains (the precisions), free to move.  The
prediction matrices and gains are the mechanism.  With the gains fixed at zero the
states never change and the model can only emit a constant prediction (the null arm);
with K = 0 there is no inference at all; K = 1 against K = 3 asks whether iterating the
inference is load-bearing, the question FEDN and NEN never answered.

Arms (two seeds):
    pc_k3            three inference steps per byte                   (the mechanism)
    pc_k1            one inference step per byte
    pc_g0            gains fixed at zero: the states never move          (a unigram model by construction)
    pc_nomem         states reset every byte: inference, no memory     (the bigram-level null arm)
    gru              a trained GRU with the same total state size      (the incumbent)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import harness
from harness import VOCAB, Budget, Bytes, TinyGRU, n_params, parse_common, train


class PredictiveCodingLM(nn.Module):
    def __init__(self, e=64, d=192, K=3, gains_trainable=True, no_memory=False):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, e)
        self.W1 = nn.Parameter(torch.randn(e, d) / math.sqrt(d))   # z1 -> predicted embedding
        self.W2 = nn.Parameter(torch.randn(d, d) / math.sqrt(d))   # z2 -> predicted z1
        g = torch.full((2, d), -1.0)
        if gains_trainable:
            self.gain_logit = nn.Parameter(g)
        else:
            self.register_buffer("gain_logit", torch.full((2, d), -30.0))   # sigmoid ~ 0
        self.K = K
        self.decay = nn.Parameter(torch.full((2, d), 2.0))        # sigmoid(decay) ~ 0.88: how much state carries over
        self.no_memory = no_memory                                 # the bigram-level null: states reset every byte
        self.readout = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, VOCAB))
        self.d = d

    def forward(self, x):
        E = self.emb(x)                                            # (B, T, e)
        B, T, _ = E.shape
        z1 = torch.zeros(B, self.d); z2 = torch.zeros(B, self.d)
        g = torch.sigmoid(self.gain_logit); dec = torch.sigmoid(self.decay)
        outs = []
        for t in range(T):
            if self.no_memory:
                z1 = torch.zeros_like(z1); z2 = torch.zeros_like(z2)
            else:
                z1 = z1 * dec[0]; z2 = z2 * dec[1]
            for _ in range(self.K):
                e1 = E[:, t] - z1 @ self.W1.T                       # (B, e)
                e2 = z1 - z2 @ self.W2.T                            # (B, d)
                z1 = z1 + g[0] * (e1 @ self.W1 - e2)
                z2 = z2 + g[1] * (e2 @ self.W2)
            outs.append(torch.cat([z1, z2], dim=-1))
        return self.readout(torch.stack(outs, dim=1))


ARMS = {
    "pc_k3": lambda d: PredictiveCodingLM(K=3),
    "pc_k1": lambda d: PredictiveCodingLM(K=1),
    "pc_g0": lambda d: PredictiveCodingLM(K=3, gains_trainable=False),
    "pc_nomem": lambda d: PredictiveCodingLM(K=3, no_memory=True),   # states reset every byte: the bigram-level null
    "gru":   lambda d: TinyGRU(64, hidden=384),
}
IS_MECH = lambda name: name in ("W1", "W2", "gain_logit", "decay")

if __name__ == "__main__":
    a = parse_common(["--ctx", "64", "--batch", "64"] + __import__("sys").argv[1:])
    harness.set_ctx(a.ctx)
    data = Bytes()
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in arms:
            if (harness.RUNS / "e6_predcode" / f"{arm}{a.suffix}_s{seed}.json").exists():
                print(f"{arm} s{seed}: done, skipping", flush=True); continue
            torch.manual_seed(seed)
            m = ARMS[arm](a.d)
            print(f"{arm}: {n_params(m):,} trainable params", flush=True)
            train(m, data, Budget(bytes=a.budget, batch=a.batch, lr=a.lr), seed, arm + a.suffix, "e6_predcode",
                  IS_MECH, extra={"params": n_params(m)})
