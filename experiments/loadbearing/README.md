# The null arm, run

The audit's cross-cutting finding was that the novel mechanism in most of the sixteen
programmes owned a sliver of its model and, in sixteen instances, never reached a
gradient. These experiments turn that around: for each mechanism, a version of the
model in which the mechanism is the load-bearing part, trained beside its null arm
(the mechanism switched off, everything else the same) and beside the incumbent it
was meant to replace, at the same byte budget, on the same held-out bytes.

Small by design: byte-level TinyStories, models of 13K to 870K parameters, 6 or 8
million training bytes per run, two seeds, two CPU cores. The question each one
answers is not "does this beat a transformer" but "when this mechanism has to carry
the model, what does it carry?"

| | mechanism | lineage | arms (mechanism / null / incumbent, then the rest) |
|---|---|---|---|
| E1 | a shared cellular-automaton rule as the whole model | DPNCA-LM, NCALM | rule iterated 6× / centre tap only (nothing crosses positions) / 2-layer transformer; also: rule applied once, 6 unshared rules, 1-layer transformer |
| E2 | a fixed random reservoir as the only sequence operator | Chaotic Reservoir LM | ρ 0.9 / ρ 0 (no recurrence) / trained GRU; also: ρ 2.1 |
| E3 | every weight emitted by a small genome | DRN | trained genome / the same phenotype with direct weights / a small direct phenotype at the genome's parameter count; also: wider genome, frozen genome with a trained readout |
| E4 | coupled oscillators as the recurrence | DRN | trained coupling / phases reset every byte (no memory) / GRU; also: fixed random coupling, coupling off |
| E5 | a morphogen field as the only thing that crosses positions | NCALM v13 | learned diffusion / diffusion off / 1-layer transformer; also: diffusion without reaction |
| E6 | hierarchical predictive coding as the whole model | APCN-LM | 3 inference steps / states reset every byte (no memory) / GRU; also: 1 step, gains at zero |

A null arm has to be able to represent a bigram model, or it says nothing about the
mechanism; E4's "coupling off" and E6's "gains at zero" arms cannot, and are kept
only as what they are (integration without interaction; a constant prediction).

**Floors.** `harness.py floors` scores interpolated n-gram models (orders 1 to 5,
Witten-Bell smoothing, no tuned weights) on the same held-out bytes, with counts from
as many training bytes as the models see (6, 8 or 24 million) and from the whole
60-million-byte slice. They calibrate every number: a model at the bigram floor has
learned nothing about context; a model under the 5-gram floor has learned more than
five bytes of memorised statistics can give.

**Checks.** `run_lrcheck.sh`, `run_sweep2.sh` and `run_sweep3.sh` sweep the learning
rate of each mechanism and its incumbent from 0.001 to 0.016 (to 0.032 where the best
was still at the top), one seed, because doubling the default rate improved every arm;
`run_sweep3.sh` then runs the two headline mechanisms and the two-layer transformer at
three times the byte budget, each at its best rate.

`harness.py` holds the data, training loop, evaluation, floors and the parameter-share
accounting; each `e*.py` is one experiment; `report.py` renders `RESULTS.md` and
`summary.json` from `runs/`, and every number in them is a run record. The data slice
is the first 60,000,000 bytes of the TinyStories training text (sha256 `ab9f3617…`)
and 1,597,917 bytes of its validation text (`c7e34622…`); place them at
`data/train.txt` and `data/val.txt`.

```bash
python harness.py floors
./run_queue.sh                                      # E1 to E6, two seeds; resumes, skipping finished arms
./run_lrcheck.sh                                    # learning rates 0.001 and 0.004, one seed
./run_extra.sh; ./run_extra2.sh; ./run_extra3.sh    # wider genome, low-rate genome, no-memory null arms
./run_sweep2.sh; ./run_sweep3.sh                    # learning rates 0.008 to 0.032, the transformer sweep, the budget check
python report.py
```

Each script waits for the ones before it and skips any arm whose run record exists, so
the whole set can be launched at once (`launch_rest.sh` detaches them).
