# Thesis LaTeX project

Working title: *Curriculum versus Mixed-Batch Training in Multi-Agent
Reinforcement Learning for Urban Autonomous Driving* (EN, triennale, ~60 pp).
Master plan: `../THESIS_OUTLINE.md` (chapter budgets, campaign tracker, open items).

## Build

```bash
latexmk -pdf -interaction=nonstopmode main.tex   # requires biber for the bibliography
```

No local TeX installation? Zip this folder and upload it to Overleaf
(compiler: pdfLaTeX; biber is supported out of the box).

## Conventions

- One file per chapter in `chapters/`; figures in `figures/`.
- `\todo{...}` marks pending content (renders red) — grep for `\todo` before submission.
- Every quantitative claim must be recomputed from `episodes.jsonl` on disk and
  traceable to a run ID (project measurement rules apply to the thesis too).
- Figure-generation scripts live in the repo (referenced in Appendix D), not ad hoc.
- `bibliography.bib` entries are pending verification (outline open item 5):
  verify each against the published version before first citation.

## Status (2026-07-14)

- Scaffold complete: main + 7 chapters + appendices + bib.
- Ch. 3: FULL DRAFT §3.1–3.7 (Tab. 1 obs, Tab. 2 rewards, Fig. 1 TikZ);
  pending figures 2–4 (`\todo` markers).
- Ch. 4: FULL DRAFT §4.1–4.5 (Tab. 3 metric catalogue, Tab. 4 condensed
  candidate registry); final `\todo`: run matrix once the campaign closes.
- Ch. 2: FULL DRAFT §2.1–2.5; all 12 bibliography entries verified against
  publisher pages on 2026-07-14 (no TODO left in bibliography.bib).
- Ch. 5: §5.1–5.4 + §5.6 FULL DRAFT — MLP half is final (recomputed via
  ../scripts/consolidate_campaign.py → ../data/campaign_consolidation.json);
  GNN-B and the two GNN evals slot in via `\todo` marks; figures:
  pilot_collapse_timeseries.pdf + learning_curves.pdf (re-run the script
  to add GNN-B). §5.5/5.7/5.8/5.9 pending.
- Ch. 6: §6.3 (threats) + §6.4 (scope) FULL DRAFT; §6.1–6.2 await Ch. 5.
- Ch. 3: Fig. 4 (curriculum state machine, TikZ) done with real unlock
  timings; Fig. 3 generator script ready in ../scripts/ (needs live CARLA,
  user runs it); Fig. 2 (Town03 screenshot) still pending.
- Sanity checks pass on all sources (env/brace balance, cross-refs, cite
  keys) — not yet compiled (no local TeX).
- Ch. 1: draft exists only in the 2026-07-06 session chat — paste into
  `chapters/ch01_introduction.tex`.
- Appendices: A (condensed registry, longtable), B (verbatim YAML snapshots
  in `configs/`), D (reproducibility) FULL DRAFT; C awaits the campaign.
- Awaiting campaign (GNN-B in flight, then evals) for Ch. 5 core, §6.1–6.2,
  Ch. 7, Abstract, App. C. Pending user actions: paste Ch. 1 draft; first
  compile on Overleaf/MiKTeX; run Fig. 3 script; Fig. 2 screenshot.
