# Data behind the write-up

| File | Contents |
|---|---|
| `scores.csv` | the sixteen audited programmes × fifteen facets (1–10), with the composite and the five axis means. Rows are keyed by the programme's source directory name. |
| `projects.json` | one record per programme (the sixteen audited plus the two after the audit): what it is, scale, what was measured, the baseline, the verdict, the outcome class, dates, and the public repository where one exists. |
| `curves.json` | the series plotted on the site: DPNCA-LM's logged validation curve, NCALM's 2.0B and 250M evaluation curves, WaveLM's ladder and ablation deltas, HDC-LM's MQAR configurations, and developmental-lm's headline numbers. Each series is copied from the run artifacts in the corresponding repository. |
| `inert.json` | the sixteen inert-mechanism instances. |
| `component_share.json` | the parameter share of the novel component in nine models. |

`scripts/make_figures.py` renders `figures/` from these files; `scripts/build_site.py` builds `docs/index.html` from them and `site/index.template.html`.
