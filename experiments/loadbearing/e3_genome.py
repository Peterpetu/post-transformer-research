"""E3: a language model whose every weight is emitted by a small genome.

DRN generated ~136M operational weights from a 418K-parameter hypernetwork over
Gray-coded index pairs, and never trained: the generator's output layer was
zero-initialised, so the model sat at ln(50257) for every step it ran.  The genome
was 0.26% of the parameters.  Here the genome is the only trainable thing.

Phenotype: the NCA-only model of E1 (one shared rule, 4 iterations, d = 64).  Every
weight tensor W of shape (m, n) is produced, once per training step, by

    W[i, j] = g([gray(i), gray(j), e_W])

where gray(.) is the Gray code of the index as a +/-1 vector, e_W is a learned
embedding of the tensor, and g is a small MLP (the genome).  Gradients flow from the
language-model loss through the phenotype into g, which is what DRN was built to do.

Arms (two seeds):
    genome           the genome trained end to end                   (the mechanism: 100% of trainable parameters)
    genome_h192      a wider genome, 44K parameters for 66K weights       (is it the compression ratio or the function class?)
    genome_frozen    the genome frozen at initialisation, a separate trainable readout  (a random-features probe)
    direct           the same phenotype with its weights trained directly  (the null arm: no genome)
    direct_small     a directly trained phenotype with about as many trainable parameters as the genome  (the incumbent at matched size)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import harness
from harness import VOCAB, Budget, Bytes, n_params, parse_common, train

BITS = 10  # enough for indices up to 1023


def gray_codes(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    g = idx ^ (idx >> 1)
    bits = ((g.unsqueeze(1) >> torch.arange(BITS)) & 1).float()
    return bits * 2 - 1  # (n, BITS) in {-1, +1}


class Genome(nn.Module):
    """g: [gray(i), gray(j), e_W] -> scalar weight."""
    def __init__(self, shapes: dict[str, tuple[int, int]], hidden=96, emb=16):
        super().__init__()
        self.shapes = shapes
        self.names = list(shapes)
        self.tensor_emb = nn.Parameter(torch.randn(len(shapes), emb) * 0.5)
        self.net = nn.Sequential(nn.Linear(2 * BITS + emb, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.scale = nn.Parameter(torch.zeros(len(shapes)))   # per-tensor log-gain
        for k, (m, n) in shapes.items():
            gi, gj = gray_codes(m), gray_codes(n)
            self.register_buffer(f"codes_{k}", torch.cat([gi.unsqueeze(1).expand(m, n, BITS), gj.unsqueeze(0).expand(m, n, BITS)], dim=-1).reshape(m * n, 2 * BITS))

    def forward(self) -> dict[str, torch.Tensor]:
        out = {}
        for t, k in enumerate(self.names):
            m, n = self.shapes[k]
            codes = getattr(self, f"codes_{k}")
            inp = torch.cat([codes, self.tensor_emb[t].expand(codes.shape[0], -1)], dim=-1)
            w = self.net(inp).reshape(m, n) * torch.exp(self.scale[t]) / math.sqrt(n)
            out[k] = w
        return out


class Phenotype(nn.Module):
    """The E1 NCA-only model with its weights supplied from outside (functional form)."""
    def __init__(self, d=64, steps=4):
        super().__init__()
        self.d, self.steps = d, steps
        self.shapes = {"emb": (VOCAB, d), "k3": (d, 3), "k5": (d, 5), "k7": (d, 7),
                       "mix1": (2 * d, 3 * d), "mix2": (d, 2 * d), "out": (VOCAB, d)}
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.step_size = nn.Parameter(torch.full((steps,), 0.5))

    def forward(self, x, W: dict[str, torch.Tensor]):
        h = F.embedding(x, W["emb"])
        d = self.d
        for s in range(self.steps):
            xt = h.transpose(1, 2)
            per = [F.conv1d(F.pad(xt, (k - 1, 0)), W[f"k{k}"].unsqueeze(1), groups=d) for k in (3, 5, 7)]
            p = torch.cat(per, dim=1).transpose(1, 2)
            m = F.linear(F.gelu(F.linear(p, W["mix1"])), W["mix2"])
            h = self.norm(h + self.step_size[s] * m)
        return F.linear(h, W["out"])


class GenomeLM(nn.Module):
    def __init__(self, d=64, steps=4, hidden=96, frozen=False):
        super().__init__()
        self.pheno = Phenotype(d, steps)
        self.genome = Genome(self.pheno.shapes, hidden=hidden)
        self.frozen = frozen
        if frozen:
            for p in self.genome.parameters():
                p.requires_grad_(False)
            self.readout = nn.Linear(d, VOCAB)   # the only trainable part of the null arm

    def forward(self, x):
        W = self.genome()
        if self.frozen:
            W = {k: v.detach() for k, v in W.items()}
            h = F.embedding(x, W["emb"]); d = self.pheno.d
            for s in range(self.pheno.steps):
                xt = h.transpose(1, 2)
                per = [F.conv1d(F.pad(xt, (k - 1, 0)), W[f"k{k}"].unsqueeze(1), groups=d) for k in (3, 5, 7)]
                p = torch.cat(per, dim=1).transpose(1, 2)
                m = F.linear(F.gelu(F.linear(p, W["mix1"])), W["mix2"])
                h = self.pheno.norm(h + self.pheno.step_size[s].detach() * m)
            return self.readout(h)
        return self.pheno(x, W)


class DirectLM(nn.Module):
    """The phenotype with ordinary, directly trained weights."""
    def __init__(self, d=64, steps=4):
        super().__init__()
        self.pheno = Phenotype(d, steps)
        self.W = nn.ParameterDict()
        for k, (m, n) in self.pheno.shapes.items():
            w = torch.randn(m, n) / math.sqrt(n)
            self.W[k] = nn.Parameter(w)

    def forward(self, x):
        return self.pheno(x, dict(self.W))


ARMS = {
    "genome":        lambda d: GenomeLM(d=64, steps=4, hidden=96),
    "genome_frozen": lambda d: GenomeLM(d=64, steps=4, hidden=96, frozen=True),
    "genome_h192":   lambda d: GenomeLM(d=64, steps=4, hidden=192),       # a genome with 2/3 as many parameters as the weights it emits
    "direct":        lambda d: DirectLM(d=64, steps=4),
    "direct_small":  lambda d: DirectLM(d=24, steps=4),
}
IS_MECH = lambda name: name.startswith("genome.")

if __name__ == "__main__":
    a = parse_common()
    harness.set_ctx(a.ctx)
    data = Bytes()
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in arms:
            if (harness.RUNS / "e3_genome" / f"{arm}{a.suffix}_s{seed}.json").exists():
                print(f"{arm} s{seed}: done, skipping", flush=True); continue
            torch.manual_seed(seed)
            m = ARMS[arm](a.d)
            gen = sum(v[0] * v[1] for v in m.pheno.shapes.values())
            print(f"{arm}: {n_params(m):,} trainable params; phenotype weights {gen:,}", flush=True)
            train(m, data, Budget(bytes=a.budget, batch=a.batch, lr=a.lr), seed, arm + a.suffix, "e3_genome",
                  IS_MECH, extra={"params": n_params(m), "phenotype_weights": int(gen)})
