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
- ALL BODY TEXT DRAFTED (2026-07-16): Ch. 2–7 complete, Abstract written,
  Appendices A–D complete. Campaign 2×2 closed and consolidated
  (../data/campaign_consolidation.json; rebuild via
  ../scripts/consolidate_campaign.py). Figures generated:
  pilot_collapse_timeseries.pdf + learning_curves.pdf (4 cells).
- Sanity checks pass on all sources (env/brace balance, cross-refs, cite
  keys) — NOT YET COMPILED (no local TeX): first Overleaf compile is the
  main outstanding validation.
- Remaining: Ch. 1 (paste user's 2026-07-06 draft, then polish + add H3);
  Fig. 2 (Town03 screenshot, user); Fig. 3 (run ../scripts/
  make_fig3_route_examples.py with live CARLA, user); optional extra
  figures (failure-mode stacked bars, Town05 bars); final \todo sweep;
  title-page fields; page-count check (~60 pp target).
