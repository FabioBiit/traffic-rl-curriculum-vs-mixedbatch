# Thesis Outline — *Curriculum vs Mixed-Batch Training in Multi-Agent RL for Urban Autonomous Driving*

> Working title. Creata il 2026-07-06 dalla sessione di pianificazione. Aggiornare questo file man mano che i capitoli vengono scritti (vedi Changelog in fondo).

## Vincoli e decisioni

- **Lunghezza**: ~60 pagine (front matter e appendici escluse dove possibile). Il budget dei capitoli somma a ~59–62 (tabelle per-cella complete in App. C).
- **Lingua**: inglese. **Livello**: triennale.
- **Campagna finale (design deciso 2026-07-08)**: matrice fattoriale **2×2×3** — {MLP, GNN critic} × {curriculum, batch} × 3 seed = 12 run da 3M, **seed appaiati su tutte le 4 celle** (999 + S2 + S3 da scegliere), stesso trunk. Contrasto primario: curriculum-vs-batch entro architettura (replicato su 2 architetture); contrasto secondario: MLP-vs-GNN entro regime; lettura d'interazione arch×regime. Vedi *Campaign tracker* in fondo.
- **GNN — due ruoli distinti**: (a) i run pilota di aprile 2026 (era pre-bugfix, non comparabili) restano **studio preliminare** (§5.1); (b) la GNN dell'era attuale (`use_gnn=true` sul trunk corrente) è **asse sperimentale secondario** della campagna 2×2×3 (§5.7). In entrambi i casi la GNN è una variante dell'encoder del **critico centralizzato** (CTDE, solo training-side); gli actor restano MLP 256×2 — in tesi va descritta così, mai come "seconda rete a confronto".
- **Framing**: il confronto curriculum-vs-mixed-batch è la spina dorsale sperimentale; la **parallel task coverage** (6 agenti su 6 rotte distinte nello stesso mondo condiviso) è la cornice concettuale, presente in *entrambi* i bracci — architettura, non variabile sperimentale.
- **Misura**: successo = `termination_reason == "route_complete"` only; veicoli e pedoni sempre riportati separatamente (le measurement rules del progetto valgono anche in tesi).

## Stato capitoli

| Capitolo | Stato |
|----------|-------|
| 1 Introduction | bozza esistente (sessione 2026-07-06, in chat) — da collocare e rifinire (aggiungere H3) |
| 2 Background | non iniziato |
| 3 System Design | non iniziato |
| 4 Experimental Methodology | non iniziato |
| 5 Results | in attesa campagna 2×2×3 (training 3/12, eval 2/12 — vedi Campaign tracker) |
| 6 Discussion | non iniziato |
| 7 Conclusions & Future Work | non iniziato |
| Appendici A–D | non iniziate |

---

## Front matter (fuori conteggio)

Title page · Abstract EN (~250 parole: domanda, metodo, 2–3 numeri chiave, conclusione — si scrive per ultimo) · TOC · Lista acronimi (RL, MARL, MAPPO, CTDE, GAE, GNN, SR, …)

## Ch. 1 — Introduction (4–5 pp)

| § | Contenuto | pp |
|---|-----------|----|
| 1.1 | Motivation: la strategia di training come decisione di design in RL; guida urbana come banco di prova | ~1 |
| 1.2 | Research question + ipotesi falsificabili: **H1** curriculum → migliore sample efficiency iniziale, rischio plateau/forgetting su hard; **H2** mixed → copertura completa da subito, segnale iniziale più rumoroso; **H3** (secondaria) entità/direzione dell'effetto del regime possono dipendere dall'encoder del critico (interazione architettura×regime). I 4 assi di confronto | ~1 |
| 1.3 | Approach at a glance: CARLA 0.9.16, MAPPO/CTDE, 6 agenti in mondo condiviso, *parallel task coverage*, campagna 2×2×3 da 3M step, Town03→Town05 | ~1 |
| 1.4 | Contributions: (a) framework di confronto a parità di budget; (b) metodologia gate-driven con registro esperimenti; (c) caratterizzazione empirica su 4 assi; (d) asse architetturale MLP/GNN critic | ~0.5 |
| 1.5 | Thesis outline | ~0.5 |

*Materiale*: bozza già scritta (sessione 2026-07-06); numeri chiave da inserire a campagna conclusa.

## Ch. 2 — Background (8–10 pp)

| § | Contenuto | pp |
|---|-----------|----|
| 2.1 | RL essentials: MDP, policy/value, policy gradient — solo ciò che serve dopo | 1.5–2 |
| 2.2 | PPO: clipped surrogate, GAE, entropy bonus (prepara lo schedule di cap. 3) | ~1.5 |
| 2.3 | MARL: non-stazionarietà, credit assignment → CTDE → MAPPO; parameter sharing; perché 2 policy separate (agenti eterogenei); cenno agli encoder del critico (MLP vs message-passing/GNN) | 2–2.5 |
| 2.4 | Curriculum learning: origini (Bengio et al. 2009), curriculum in RL (survey Narvekar et al. 2020), curricula automatici; il mixed/interleaved come controllo "no ordering"; catastrophic forgetting | ~2 |
| 2.5 | CARLA: 0.9.16 su UE4, synchronous mode, Traffic Manager; caratteristiche Town03 vs Town05 | 1–1.5 |

*Nota bibliografia*: raccogliere 20–30 riferimenti; ogni claim di letteratura con fonte concreta (regola del progetto: mai "state of the art" senza fonte). Citazioni indicate **da verificare**.

## Ch. 3 — System Design (11–13 pp) — *il progetto*

| § | Contenuto | pp |
|---|-----------|----|
| 3.1 | Architecture overview: CARLA server ↔ env multi-agente ↔ Ray/RLlib 2.10 MAPPO ↔ logging `episodes.jsonl` ↔ eval pipeline. **Fig. 1**: diagramma componenti | ~1.5 |
| 3.2 | Shared-world multi-agent environment: 6 agenti (3 veicoli + 3 pedoni), rotte per-agente indipendenti e seedate (`SeedSequence[traffic_seed, reset_count, crc32(agent_id)]`), termination reasons (route_complete/collision/offroad/stuck/timeout), 6 record/episodio. Qui si formalizza la **parallel task coverage** + caveat (campioni correlati, non-stazionarietà). **Fig. 2**: screenshot Town03 con 6 rotte | 2–2.5 |
| 3.3 | Observation & action spaces: veicolo 47D (feature route-aware; razionale anti state-aliasing di O1+O2), pedone 26D, canale hazard (`ped_ttc`/`ped_occ` → `hazard_risk`), global obs 225D per il critico; azioni continue 2D per classe — veicolo `[throttle/brake unificato, steer]` ∈ [−1,1]², pedone `[speed_frac ∈ [0,1] (×5.0 m/s max), Δheading ∈ [−1,1] (±45°/step)]`. **Tab. 1**: componenti obs | ~2 |
| 3.4 | Reward design: veicolo (progress, `target_min_speed=8` km/h, gate `safe_to_push` hazard<0.85, penalità no_wp>100, collision −500, offroad) e pedone (progress, comfort band [1.2, 2.6] m/s); razionale per componente, evoluzione → rimando cap. 4. **Tab. 2**: componenti reward con pesi | 2–2.5 |
| 3.5 | Route generation & difficulty: route_planner, distanze target 30/60/100 m (`levels.yaml`), vincolo lunghezza ≤2× target, crosswalk `max_cross=3`, tassonomia route_source (legacy/sidewalk_fallback) e feasibility pedonale. **Fig. 3**: esempi di rotte per livello | 1.5–2 |
| 3.6 | **Curriculum vs batch mode** (sezione chiave): curriculum manager — unlock a SR windowed bilanciata (70% medium / 60% hard), `min_budget_share`, force-unlock caps (30%/70% del budget), budget shares 0.30/0.35/0.35, probation; batch — `BatchLevelSampler`: *stratified shuffle without replacement*, ogni finestra di 3 episodi è una permutazione casuale di easy/medium/hard → esposizione uniforme 1/3 per livello da step 0, sbilanciamento max 1 episodio. **Fig. 4**: state machine del curriculum + timeline unlock di esempio | ~2 |
| 3.7 | Critic encoder variants (asse architetturale): MLP (default, hidden 256×2) vs `GNNCriticEncoder` (message passing sul grafo dei 6 agenti: embed 64, 2 layer, 4 head; solo critico, actor invariati). Training & eval pipeline: γ=0.997, entropy schedule fraction-based 0.03→0.005 @83%, vf settings, 3M budget, checkpointing; static eval vs final eval (4 scenari × 100 ep, incl. Town05); integrità log | 1.5–2 |

*Fonti*: `train_mappo.yaml`, `curriculum_batch.yaml`, `levels.yaml`, `carla_multi_agent_env.py`, `route_planner.py`, `curriculum_batch_manager.py`, `centralized_critic.py`.

## Ch. 4 — Experimental Methodology (8–9 pp) — *materiale distintivo*

| § | Contenuto | pp |
|---|-----------|----|
| 4.1 | Metrics & measurement rules: successo = `route_complete` only; aggregazione cumulativa agent-level; dedup `episode_id+agent_id`; sempre 3 viste (all/veh/ped). **Tab. 3**: catalogo metriche (SR, stuck, timeout, s+t, collision, offroad, route completion, path efficiency, speed, no_wp_steps) | 1.5–2 |
| 4.2 | Gate-driven development protocol: A/B single-knob, seed-pairing, 300K per candidato, gate (+2pp SR, −2pp s+t, ≤+1pp coll/off, integrità 6/6, no NaN/inf), regola di revert. Perché: varianza alta + budget limitato | 1.5–2 |
| 4.3 | **Environment validity — bugs found & fixed** (sezione "honest science"): route-len bugfix (~+50pp SR = cambio distribuzione task → non-comparabilità pre/post), route-seed fix (pairing A/B), bug pipeline eval (seed collapse, deterministic policy, JSONL mancante). Lezione: validità della misura prima del tuning | ~2 |
| 4.4 | Iterative development summary: fasi (pilot architetturale aprile → fix correttezza → obs 47D → reward shaping D2/R1/V1 → crosswalk max_cross=3) con esito gate in 1–2 righe ciascuna; registro completo → App. A. **Tab. 4**: candidati principali con esito | 1.5–2 |
| 4.5 | Final campaign design: **matrice 2×2×3** — {MLP, GNN critic} × {curriculum unlock-path, batch interleaved} × 3 seed appaiati — 12 run 3M sullo stesso trunk, difficoltà=path; contrasto primario C-vs-B entro architettura, secondario MLP-vs-GNN entro regime, lettura d'interazione; eval a 3 livelli (training cumulativa + static + stochastic final incl. Town05). Dichiarare cosa NON si testa (traffic/mixed difficulty, attention/PopArt, actor non-MLP) | 1–1.5 |

## Ch. 5 — Results (14–16 pp)

| § | Contenuto | pp |
|---|-----------|----|
| 5.1 | Preliminary architectural pilot: 9 run di aprile 2026 (MLP/GNN × C/B, era pre-route-len-bugfix). Audit 2026-07-08: dati integri e presentabili, ma **veicoli a floor in tutte le celle (SR 0.2–3.8%)** → il confronto architetturale sui veicoli non è informativo; presentare tabella compatta + lettura qualitativa (GNN-curr profilo più difensivo: s+t 70.7% / coll 19.5% vs MLP-curr 53.1% / 30.6%; il collasso del run 0504 NON è visibile nel cumulativo → mostrarlo con le serie temporali). Aggancio narrativo a §4.3: il floor veicoli fu poi spiegato dal route-len bug | 1–1.5 |
| 5.2 | Campaign integrity & setup recap: conteggi episodi, no NaN/inf, timesteps, commit hash per run. **Tab. 5**: run matrix (12 run: 2 arch × 2 regimi × 3 seed) | 0.5–1 |
| 5.3 | **Sample efficiency**: learning curve SR veicoli e pedoni vs steps per cella (media ± range su 3 seed); time-to-threshold; timing unlock del curriculum (primo ep medium/hard). **Figg. 5–6** | 2.5–3 |
| 5.4 | **Final cumulative performance**: tabella headline cella (arch×regime) × classe × metrica; composizione failure mode (stacked bars); breakdown per livello easy/medium/hard. **Figg. 7–8, Tab. 6** | 2.5–3 |
| 5.5 | Behavioral analysis: traiettorie Q1→Q4, profili velocità, no_wp_steps, stuck causes; caso studio *defensive equilibrium* (gate hazard) + accoppiamento P1×V1 come evidenza di interazione tra classi | 2–2.5 |
| 5.6 | **Generalization Town03→Town05**: metriche test per cella, generalization gap (drop SR), per classe — il test più diretto dell'ipotesi "copertura → robustezza". **Fig. 9** | 2–2.5 |
| 5.7 | **Architecture axis (secondario, H3)**: MLP vs GNN critic entro regime; il contrasto C-vs-B si replica su entrambe le architetture? Interazione arch×regime; aggancio al pilot §5.1 (la firma difensiva GNN-curr di aprile si ripresenta sul trunk attuale?) | 1.5–2 |
| 5.8 | Pedestrian structural limits: feasibility rotte (under_target/too_short/sidewalk_fallback), collasso medium/hard: separare limiti ambiente da limiti policy | 1–1.5 |
| 5.9 | Findings vs hypotheses: tabella H1/H2/H3 → supportata/respinta/parziale per asse | 0.5–1 |

## Ch. 6 — Discussion (4–5 pp)

- 6.1 Cosa cambia (e cosa no) con l'ordine dei task: interpretazione dei 4 assi.
- 6.2 Parallel task coverage rivisitata: cosa dice l'evidenza su interazioni, campioni correlati, non-stazionarietà.
- 6.3 Threats to validity: non-determinismo CARLA same-seed (range SR 52–62% documentato su repliche identiche), n=3 per cella, singola coppia di mappe, accoppiamento reward-curriculum, lezioni della pipeline eval.
- 6.4 Scope limitations: actor architecture fissa (varia solo l'encoder del critico), nessuna ablazione 1-vs-6 agenti, generazione rotte pedoni.

## Ch. 7 — Conclusions & Future Work (2–3 pp)

- 7.1 Risposta alla research question + contributi.
- 7.2 Future work: estensioni architetturali (GAT = GNN+attention, PopArt — flag già in infrastruttura), scaling 100+ agenti (visione applicativa), curricula adattivi/automatici, fix generazione rotte pedonali.

## Appendices (3–5 pp)

- **A** Experiment registry (condensato da `docs/EXPERIMENT_REGISTRY.md`)
- **B** Iperparametri completi (`train_mappo.yaml`, `curriculum_batch.yaml`, `levels.yaml`)
- **C** Risultati per-seed (tabelle complete dei 12 run)
- **D** Reproducibility: comandi, seed, commit hash, hardware

---

## Ordine di scrittura consigliato

**Ch. 3 → Ch. 4 → Ch. 2 → Ch. 5 → Ch. 6–7 → rifinitura Ch. 1 → Abstract.**
I capitoli 3 e 4 sono scrivibili subito (tutto documentato nel repo, indipendenti dai run finali); il 2 si scrive mentre gira la campagna; il 5 appena la campagna è consolidata.

## Campaign tracker 2×2×3 (aggiornato 2026-07-08)

| Cella | Seed 999 | Seed S2 (TBD) | Seed S3 (TBD) |
|-------|----------|---------------|---------------|
| MLP × curriculum | train+eval OK — `20260622_171626` | — | — |
| MLP × batch | train+eval OK — `20260623_171855` | — | — |
| GNN × curriculum | train OK, **eval pending** — `20260630_181143` | — | — |
| GNN × batch | **mancante** | — | — |

Rimanenti: 9 training 3M + 10 eval. Priorità consigliata: (1) eval GNN-curr 999; (2) GNN-batch 999 train+eval → completa la matrice n=1 e sblocca la stesura del cap. 5; (3) repliche S2/S3. I run li lancia sempre l'utente (mai Claude).

## Open items (da chiudere prima delle sezioni corrispondenti)

- [x] 1. **CHIUSO 2026-07-08** — Audit `episodes.jsonl` run di aprile: 9 run, tutti integri (6 rec/ep 100%, 0 bad lines, schema attuale). Veicoli a floor ovunque (SR 0.2–3.8%), pedoni 66–82%. GNN-curr 0425 profilo più difensivo (s+t 70.7%, coll 19.5%). Il run 0504 `EARLY_COLLASSO` non mostra il collasso nel cumulativo (veh SR 3.8%, il più alto dei 9) → in tesi va mostrato con le serie temporali. Run 0408 senza tag architettura nel nome (*chiedere all'utente se MLP*). Verdetto: §5.1 presentabile con numeri, come sezione cautelativa/qualitativa
- [x] 2. **CHIUSO 2026-07-08** — `BatchLevelSampler` = stratified shuffle without replacement: ogni finestra di 3 episodi è una permutazione casuale dei 3 livelli; uniforme 1/3 da step 0; sbilanciamento max 1 episodio (`curriculum_batch_manager.py:902`)
- [x] 3. **CHIUSO 2026-07-08** — Campagna identificata su disco e design deciso (**2×2×3**): MLP-C `20260622_171626` (3M, seed 999, train+eval OK, 3220 ep + 400 ep eval), MLP-B `20260623_171855` (3M, seed 999, train+eval OK, 3180 ep + 400 ep eval), GNN-C `20260630_181143` (3M, seed 999, `use_gnn=true`, train OK 3293 ep / eval pending), GNN-B mancante. I grafici attuali dell'utente si basano sulla coppia MLP. Vedi Campaign tracker
- [x] 4. **CHIUSO 2026-07-08** — Action space: veicolo Box 2D `[tb, steer]` ∈ [−1,1]² (tb>0 → throttle, tb<0 → brake; reverse fisso off); pedone Box 2D `[speed_frac, Δheading]` con speed_frac ∈ [0,1] × `PEDESTRIAN_MAX_SPEED=5.0` m/s e Δheading ∈ [−1,1] × ±45°/step (`carla_multi_agent_env.py:311-319`, `:1255-1278`)
- [ ] 5. Verifica citazioni (Bengio 2009; Narvekar et al. 2020; Yu et al. 2022; Dosovitskiy et al. 2017 per CARLA) → Ch. 2 *(insieme)*
- [ ] 6. Scelta seed S2/S3 (identici su tutte le 4 celle) *(utente)*
- [ ] 7. Lanci rimanenti: eval GNN-C 999 → GNN-B 999 (train+eval) → 8 run S2/S3 + eval *(l'utente lancia; Claude consolida i risultati)*

## Changelog

- 2026-07-06: prima versione, dalla sessione di pianificazione (decisioni: inglese, triennale, 3+3 seed appaiati, GNN come pilot §5.1 + future work).
- 2026-07-08: chiusi open item 1, 2, 4 (audit pilot aprile: dati integri ma veicoli a floor SR 0.2–3.8%; batch sampler = stratified shuffle uniforme; action space 2D+2D verificato). Aggiornate §3.3, §3.6, §5.1.
- 2026-07-08 (2): open item 3 chiuso (run campagna identificati su disco); **design campagna rivisto: 2×2×3** {MLP, GNN critic} × {C, B} × 3 seed = 12 run 3M (supera il "3+3 solo MLP" del 07-06). Aggiunti H3 (§1.2), §3.7 critic variants, §4.5 fattoriale, §5.7 architecture axis (rinumerate 5.8–5.9), Campaign tracker, open item 6–7. GNN promossa da pilot-only ad asse secondario.
