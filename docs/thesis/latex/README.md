# Thesis LaTeX project

Working title: *Curriculum versus Mixed-Batch Training in Multi-Agent
Reinforcement Learning for Urban Autonomous Driving* (EN, triennale, ~60 pp).
Master plan: `../THESIS_OUTLINE.md` (chapter budgets, campaign tracker, open items).

## Build

Local toolchain: **MiKTeX 25.12** (installed 2026-07-17, per-user, AutoInstall
on). From this folder (`$bin = %LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`,
or plain `pdflatex`/`biber` in any shell opened after the install):

Two documents share `figures/`, `configs/` and `bibliography.bib`:
`main_en.tex` (English, `chapters/`) and `main_it.tex` (Italian
translation, `chapters_it/`; keeps identical labels, numbers and
structure).

```powershell
pdflatex -interaction=nonstopmode main_en.tex
biber main_en
pdflatex -interaction=nonstopmode main_en.tex
pdflatex -interaction=nonstopmode main_en.tex
# same sequence for the Italian version: main_it
```

First full build 2026-07-17: **68 pages, no unresolved references**; only
cosmetic warning is the bold small-caps substitution on the title page
(disappears once the `\todo` placeholders are replaced). Overleaf remains an
option (upload `../thesis_overleaf_*.zip`), no longer required.

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
- Ch. 1: FULL DRAFT (2026-07-17, written fresh; merge the old 06-07 chat
  draft selectively if it ever resurfaces). Figures failure_modes.pdf +
  town05_bars.pdf generated and wired into §5.4/§5.6.
- 2026-07-19: post-review pass. Corrections applied (stale §5.3
  take-off numbers updated with GNN-B, Appendix B/D overfull lines
  fixed, config-snapshot comments re-wrapped/translated, MLP-curriculum
  caption attribution, roman-numbered front matter). `main.tex` renamed
  to `main_en.tex` (68 pp); full Italian translation added as
  `main_it.tex` + `chapters_it/` (71 pp; cleveref Italian names defined
  manually via \AtBeginDocument). Both compile clean with MiKTeX: no
  errors, no unresolved references, no overfull > 20pt.
- 2026-07-19 (bis): generic title page replaced in both documents with
  the official UniMarconi template (from Downloads/Frontespizio.doc via
  LibreOffice conversion): logo extracted to figures/logo_unimarconi.jpg,
  university heading, DIPARTIMENTO/CORSO lines, quoted italic title,
  Relatore/Candidato two-column block, ANNO ACCADEMICO.
- Remaining: Fig. 2 (Town03 screenshot, user); Fig. 3 (run ../scripts/
  make_fig3_route_examples.py with live CARLA, user); title-page blank
  fields (5 \todo each in main_en.tex and main_it.tex: dipartimento,
  corso di laurea, relatore, candidato, anno accademico); page-count
  check; language polish pass.
