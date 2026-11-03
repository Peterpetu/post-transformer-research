# Results: the null arm, run

Byte-level language models on the same TinyStories slice, scored on the same 262,144 held-out bytes (bits per byte, lower is better). Within an experiment every arm trains under the same byte budget, context length and optimiser settings; two seeds, both values shown. `share` is the fraction of the model's trainable parameters inside the mechanism under test, the quantity the audit's "the new part owns almost none of the model" chart reports.

**Floors.** Interpolated n-gram models (Witten-Bell smoothing, no tuned weights) scored on the same bytes, with counts from as many training bytes as the models in that experiment see. They say what memorising short-range statistics is worth here; a mechanism that lands at the bigram floor has learned nothing about context. With the whole 60,000,000-byte slice the floors barely move (unigram 4.387, bigram 3.314, trigram 2.390, 4-gram 1.805, 5-gram 1.513), so the data, not the budget, sets them. The n-gram sees the evaluation stream contiguously while the models see fixed windows, which makes the floors slightly stricter than a windowed n-gram would reach.

## E1. A language model that is nothing but a cellular automaton

Lineage: DPNCA-LM, NCALM. Share of the model the mechanism held there: 0.083% (DPNCA-LM's cross-position mixing). What crosses positions here: the rule's depthwise perception (kernels 3, 5, 7), iterated. Budget 8,000,000 bytes, context 256. Floors at this budget: unigram 4.390, bigram 3.317, trigram 2.397, 4-gram 1.820, 5-gram 1.537.

| arm | role | trainable params | share in mechanism | bits/byte | seeds | wall-clock |
|---|---|---|---|---|---|---|
| **nca_shared6** | one rule, iterated 6 times, weights shared (the mechanism) | 199,814 | 67.1% | **1.432** | 1.430, 1.433 | 9 min |
| nca_shared1 | the rule applied once | 199,809 | 67.1% | **1.611** | 1.608, 1.614 | 2 min |
| nca_unshared6 | 6 separate rules: a plain causal CNN | 869,894 | 92.4% | **1.266** | 1.267, 1.266 | 8 min |
| nca_local6 | perception reduced to the centre tap: nothing crosses positions (the null arm) | 199,814 | 67.1% | **3.317** | 3.317, 3.318 | 5 min |
| transformer | 2-layer transformer, d = 128 (the incumbent) | 495,360 | 0.0% | **1.834** | 1.859, 1.809 | 5 min |
| transformer_s | 1-layer transformer, d = 96 (a smaller incumbent) | 186,016 | 0.0% | **2.303** | 2.308, 2.298 | 2 min |

Mechanism against its null arm: **+1.886** bits/byte. Against the incumbent at the shared learning rate: **-0.402**; at each one's best learning rate in the sweep (seed 0): **-0.129**. The mechanism sits below the 5-gram floor (1.54). Reading: **carries the model, 0.13 bits/byte ahead of the incumbent at each one's best learning rate**.

Learning-rate sweep (seed 0; the main runs use 0.002; best in bold):

| arm | lr 0.001 | lr 0.002 | lr 0.004 | lr 0.008 | lr 0.016 | lr 0.032 | best |
|---|---|---|---|---|---|---|---|
| nca_shared6 | 1.542 | 1.430 | 1.385 | **1.366** | 1.374 | – | 1.366 at 0.008 |
| transformer | 2.421 | 1.859 | 1.576 | **1.495** | 1.505 | 1.542 | 1.495 at 0.008 |
| transformer_s | 2.911 | 2.308 | 1.893 | 1.676 | **1.614** | 1.630 | 1.614 at 0.016 |

Budget check (seed 0; three times the bytes, each arm at its best learning rate from the sweep):

| arm | learning rate | 8,000,000 bytes | 24,000,000 bytes | gain |
|---|---|---|---|---|
| nca_shared6 | 0.008 | 1.366 | 1.279 | 0.087 |
| transformer | 0.008 | 1.495 | 1.235 | 0.260 |

Floors at 24,000,000 bytes: unigram 4.387, bigram 3.313, trigram 2.389, 4-gram 1.805, 5-gram 1.516.

## E2. A fixed random reservoir as the only sequence operator

Lineage: Chaotic Reservoir LM. Share of the model the mechanism held there: 0.202% of parameters, 0% of trainable ones (the reservoir). What crosses positions here: the fixed reservoir matrix W_res, recurring over positions. Budget 6,000,000 bytes, context 64. Floors at this budget: unigram 4.390, bigram 3.318, trigram 2.400, 4-gram 1.825, 5-gram 1.547.

| arm | role | trainable params | share in mechanism | bits/byte | seeds | wall-clock |
|---|---|---|---|---|---|---|
| **reservoir_r0.9** | fixed random reservoir, spectral radius 0.9 (the mechanism) | 197,120 | 25% of all parameters, fixed | **1.889** | 1.883, 1.896 | 2 min |
| reservoir_r2.1 | spectral radius 2.1, the original project's setting | 197,120 | 25% of all parameters, fixed | **1.727** | 1.724, 1.731 | 2 min |
| reservoir_r0 | no recurrence: h_t = tanh(u_t) (the null arm) | 197,120 | 25% of all parameters, fixed | **3.317** | 3.317, 3.317 | 2 min |
| gru | a trained GRU with the same state size (the incumbent) | 395,008 | 0.0% | **1.563** | 1.567, 1.559 | 3 min |

Mechanism against its null arm: **+1.428** bits/byte. Against the incumbent at the shared learning rate: **+0.327**; at each one's best learning rate in the sweep (seed 0): **+0.202**. The mechanism sits between the trigram (2.40) and 4-gram (1.83) floors. Reading: **carries the model, 0.20 bits/byte behind the incumbent at each one's best learning rate**.

Learning-rate sweep (seed 0; the main runs use 0.002; best in bold):

| arm | lr 0.001 | lr 0.002 | lr 0.004 | lr 0.008 | lr 0.016 | lr 0.032 | best |
|---|---|---|---|---|---|---|---|
| gru | 1.686 | 1.567 | 1.486 | **1.474** | 1.534 | – | 1.474 at 0.008 |
| reservoir_r0.9 | 1.994 | 1.883 | 1.782 | 1.703 | **1.676** | 1.739 | 1.676 at 0.016 |

## E3. Every weight emitted by a small genome

Lineage: DRN. Share of the model the mechanism held there: 0.26% (the weight generator). What crosses positions here: the phenotype (E1's cellular automaton, d = 64, 4 iterations); the genome only writes its weights. Budget 8,000,000 bytes, context 256. Floors at this budget: unigram 4.390, bigram 3.317, trigram 2.397, 4-gram 1.820, 5-gram 1.537.

| arm | role | trainable params | share in mechanism | bits/byte | seeds | wall-clock |
|---|---|---|---|---|---|---|
| **genome** | 13K-parameter genome emitting all 66K phenotype weights, trained end to end (the mechanism) | 13,084 | 100.0% | **3.421** | 3.420, 3.421 | 5 min |
| genome_h192 | a wider genome, 44K parameters for the same 66K weights | 44,476 | 100.0% | **3.893** | 4.390, 3.396 (seeds disagree) | 7 min |
| genome_frozen | genome frozen at initialisation with a trained readout: a random-features probe | 16,644 | 0.0% | **3.858** | 3.970, 3.747 | 2 min |
| direct | the same phenotype with its 66K weights trained directly (the null arm: no genome) | 66,500 | 0.0% | **1.690** | 1.690, 1.690 | 2 min |
| direct_small | a directly trained phenotype with 17K parameters, about the genome's count (the incumbent at matched size) | 17,260 | 0.0% | **2.192** | 2.203, 2.181 | 1 min |

Mechanism against its null arm: **-1.730** bits/byte. Against the incumbent at the shared learning rate: **+1.229**; at each one's best learning rate in the sweep (seed 0): **+1.421**. The mechanism sits between the unigram (4.39) and bigram (3.32) floors. Reading: **worse than its null arm**.

Learning-rate sweep (seed 0; the main runs use 0.002; best in bold):

| arm | lr 0.0005 | lr 0.001 | lr 0.002 | lr 0.004 | lr 0.008 | lr 0.016 | lr 0.032 | best |
|---|---|---|---|---|---|---|---|---|
| direct | – | 1.814 | 1.690 | 1.608 | 1.565 | 1.555 | **1.552** | 1.552 at 0.032 |
| direct_small | – | 2.361 | 2.203 | 2.096 | 2.022 | 1.975 | **1.967** | 1.967 at 0.032 |
| genome_h192 | 4.391 | – | **4.390** | – | – | – | – | 4.390 at 0.002 |
| genome | 3.939 | 3.427 | 3.420 | **3.388** | 4.390 | 4.389 | – | 3.388 at 0.004 |

## E4. Coupled oscillators as the recurrence

Lineage: DRN. Share of the model the mechanism held there: 0.0016% (the Kuramoto term). What crosses positions here: the oscillator phases, carried from byte to byte and coupled through A. Budget 6,000,000 bytes, context 64. Floors at this budget: unigram 4.390, bigram 3.318, trigram 2.400, 4-gram 1.825, 5-gram 1.547.

| arm | role | trainable params | share in mechanism | bits/byte | seeds | wall-clock |
|---|---|---|---|---|---|---|
| **kuramoto** | trained coupling A, learned K (the mechanism) | 328,193 | 20.0% | **2.038** | 2.039, 2.037 | 4 min |
| kuramoto_fixedA | fixed random coupling, everything else trained | 262,657 | 0.0% | **3.922** | 3.948, 3.896 | 3 min |
| kuramoto_k0 | coupling off: phases only accumulate the input (integration without interaction; cannot represent a bigram) | 262,656 | 0.0% | **3.955** | 3.957, 3.954 | 2 min |
| kuramoto_nomem | phases reset every byte: no memory at all (the null arm) | 262,656 | 0.0% | **3.318** | 3.318, 3.319 | 2 min |
| gru | a trained GRU with the same state size (the incumbent) | 395,008 | 0.0% | **1.563** | 1.567, 1.559 | 2 min |

Mechanism against its null arm: **+1.280** bits/byte. Against the incumbent at the shared learning rate: **+0.475**; at each one's best learning rate in the sweep (seed 0): **+0.309**. The mechanism sits between the trigram (2.40) and 4-gram (1.83) floors. Reading: **carries the model, 0.31 bits/byte behind the incumbent at each one's best learning rate**.

Learning-rate sweep (seed 0; the main runs use 0.002; best in bold):

| arm | lr 0.001 | lr 0.002 | lr 0.004 | lr 0.008 | lr 0.016 | lr 0.032 | best |
|---|---|---|---|---|---|---|---|
| gru | 1.686 | 1.567 | 1.486 | **1.474** | 1.534 | – | 1.474 at 0.008 |
| kuramoto | 2.310 | 2.039 | 1.887 | 1.804 | **1.783** | 1.869 | 1.783 at 0.016 |

## E5. A morphogen field as the only thing that crosses positions

Lineage: NCALM v13. Share of the model the mechanism held there: 0.39% (the morphogen field). What crosses positions here: causal diffusion along positions: 512 learned per-channel, per-step rates. Budget 8,000,000 bytes, context 256. Floors at this budget: unigram 4.390, bigram 3.317, trigram 2.397, 4-gram 1.820, 5-gram 1.537.

| arm | role | trainable params | share in mechanism | bits/byte | seeds | wall-clock |
|---|---|---|---|---|---|---|
| **morphogen** | learned diffusion, learned reaction (the mechanism) | 331,264 | 80.1% | **1.424** | 1.426, 1.421 | 4 min |
| morphogen_nodiff | diffusion off: the reaction alone, so nothing crosses positions (the null arm) | 330,752 | 80.0% | **3.318** | 3.317, 3.318 | 2 min |
| morphogen_lin | diffusion only, no reaction: a linear model | 66,560 | 0.8% | **2.834** | 2.835, 2.832 | 2 min |
| transformer_s | 1-layer transformer, d = 96 (the incumbent) | 186,016 | 0.0% | **2.303** | 2.308, 2.298 | 2 min |

Mechanism against its null arm: **+1.894** bits/byte. Against the incumbent at the shared learning rate: **-0.879**; at each one's best learning rate in the sweep (seed 0): **-0.279**. The mechanism sits below the 5-gram floor (1.54). Reading: **carries the model, 0.28 bits/byte ahead of the incumbent at each one's best learning rate**.

Learning-rate sweep (seed 0; the main runs use 0.002; best in bold):

| arm | lr 0.001 | lr 0.002 | lr 0.004 | lr 0.008 | lr 0.016 | best |
|---|---|---|---|---|---|---|
| morphogen | 1.539 | 1.426 | 1.366 | **1.334** | 1.335 | 1.334 at 0.008 |
| transformer_s | 2.911 | 2.308 | 1.893 | 1.676 | **1.614** | 1.614 at 0.016 |

Budget check (seed 0; three times the bytes, each arm at its best learning rate from the sweep):

| arm | learning rate | 8,000,000 bytes | 24,000,000 bytes | gain |
|---|---|---|---|---|
| morphogen | 0.008 | 1.334 | 1.213 | 0.121 |

Floors at 24,000,000 bytes: unigram 4.387, bigram 3.313, trigram 2.389, 4-gram 1.805, 5-gram 1.516.

## E6. Hierarchical predictive coding as the whole model

Lineage: APCN-LM. Share of the model the mechanism held there: about 30% (the non-transformer part). What crosses positions here: two levels of latent state, carried from byte to byte and corrected by prediction errors. Budget 6,000,000 bytes, context 64. Floors at this budget: unigram 4.390, bigram 3.318, trigram 2.400, 4-gram 1.825, 5-gram 1.547.

| arm | role | trainable params | share in mechanism | bits/byte | seeds | wall-clock |
|---|---|---|---|---|---|---|
| **pc_k3** | three inference steps per byte (the mechanism) | 230,656 | 21.6% | **2.035** | 2.035, 2.035 | 5 min |
| pc_k1 | one inference step per byte | 230,656 | 21.6% | **2.021** | 2.023, 2.020 | 2 min |
| pc_g0 | gains fixed at zero: the states never move (a unigram model by construction) | 230,272 | 21.5% | **4.388** | 4.388, 4.387 | 5 min |
| pc_nomem | states reset every byte: inference without memory (the null arm) | 230,656 | 21.6% | **3.318** | 3.318, 3.318 | 4 min |
| gru | a trained GRU with the same total state size (the incumbent) | 633,344 | 0.0% | **1.479** | 1.484, 1.475 | 4 min |

Mechanism against its null arm: **+1.283** bits/byte. Against the incumbent at the shared learning rate: **+0.556**; at each one's best learning rate in the sweep (seed 0): **+0.363**. The mechanism sits between the trigram (2.40) and 4-gram (1.83) floors. Reading: **carries the model, 0.36 bits/byte behind the incumbent at each one's best learning rate**.

Learning-rate sweep (seed 0; the main runs use 0.002; best in bold):

| arm | lr 0.001 | lr 0.002 | lr 0.004 | lr 0.008 | lr 0.016 | lr 0.032 | best |
|---|---|---|---|---|---|---|---|
| gru | 1.610 | 1.484 | **1.409** | 1.414 | 1.520 | – | 1.409 at 0.004 |
| pc_k3 | 2.200 | 2.035 | 1.907 | 1.820 | **1.773** | diverged | 1.773 at 0.016 |

## Caveats

- Same budget, not convergence. Every arm is still improving when its budget ends, and the learning rate moves every arm by more than the seed spread does. Each mechanism and incumbent was therefore swept over 0.001 to 0.016 (0.032 where the best was still at the top; one seed): the comparisons with the incumbents use each arm's best rate, and the budget check at three times the bytes, at those rates, shows how the gaps move with data. For `reservoir_r0.9`, `kuramoto` and `pc_k3` the best rate was the top of the sweep (0.016; 0.032 was worse or diverged) and for `direct` 0.032 beat 0.016 by 0.003; every other best rate is interior.
- Two seeds at the default rate; one seed in the sweep and the budget check. The spread between seeds is 0.001 to 0.05 bits/byte for every arm except `genome_frozen` (0.22: a random-features probe, whose features are the draw), `kuramoto_fixedA` (0.05) and the wider genome, whose seeds are shown separately. Differences under 0.1 bits/byte are not read as differences.
- Share. `share` counts trainable parameters inside the mechanism under test. In E5 that includes the per-position reaction MLPs; only the 512 diffusion rates move information across positions, so the mechanism's claim there rests on the null arm (diffusion off), not on the share. In E2 the reservoir has no trainable parameters, so its share of all parameters is reported instead.
- Nulls. E4's `kuramoto_k0` and E6's `pc_g0` switch the mechanism off but cannot represent even a bigram model (the oscillators integrate without reading the current byte; the zero-gain states never move), so they are not the null arms. The null arms are `kuramoto_nomem` and `pc_nomem`, which keep the per-byte computation and remove memory. E3's null arm is the same phenotype with directly trained weights; `genome_frozen` is a random-features probe. The null arms were run at the default rate only: each sits within 0.01 bits/byte of the bigram floor, and a model at the bigram floor has nothing left to gain from its optimiser.
- Floors. The n-gram floors use Witten-Bell interpolation with counts from the first 6,000,000, 8,000,000 or 24,000,000 training bytes (whichever the budget is), scored on the contiguous evaluation stream. Better smoothing (for instance Kneser-Ney) would lower the higher-order floors somewhat; the bigram floor, which the reading rule uses, is insensitive to this.
- Scale. These are byte-level models of 13K to 870K parameters trained on 6 to 24 million bytes on two CPU cores. They say what each mechanism carries when it has to carry the model, and how that moves with the learning rate and the budget at this scale; they do not say where any of them would end up.

