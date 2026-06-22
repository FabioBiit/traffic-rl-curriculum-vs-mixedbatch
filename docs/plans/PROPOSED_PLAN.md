<!--
PROPOSED_PLAN.md — journal di pianificazione CARLA MAPPO (curriculum vs mixed-batch).
Compattato 17-06-2026: Q&A riordinate (recenti → vecchie), rimosse duplicazioni/verbosità/filler;
preservati TUTTI i verdetti, meccanismi, run ID e core delle Q&A.
- Track record numerico completo (tabelle A/B per candidato): CLAUDE.md <experimental_state> + Candidate Registry, AGENTS.md, docs/EXPERIMENT_REGISTRY.md.
- Copia verbatim pre-compattazione: C:\Users\kyros\AppData\Local\Temp\claude_analysis\PROPOSED_PLAN_backup_20260617.md (+ git HEAD b25e51c).
Struttura: PIANI (recente→vecchio) · TRACK RECORD · Q&A (recente→vecchio) · BUGS/EVAL.
-->

#########################################################  PIANI (recente → vecchio)  #########################################################

### [17-06-2026] P1 max_cross — crossing-frequency retune (ATTIVO, branch EVO/ped-crosswalk-p1)

**Considerazioni (locked 300K P1 vs r0610 per-livello).** P1 = trade-off: peds in forte crescita
(medium +49.6, hard +56pp) e SR veicoli **preservata** (medium +13, hard +2.7pp), ma costo di sicurezza:
**offroad universale** (+2.5–3.8pp, gate FAIL su medium E hard, in salita nella fase matura) e **collision
hard strutturale** (~17%, piatta nel mature −0.6pp; il controllo medium INVECE cala 11.2→8.3% → la collision
è imparabile dove il carico crossing è basso). Verdetto empirico (mature-phase trend): SR hard ancora in
salita (+14.3pp, stuck+TO −16.2) → NON degrada; ma collision/offroad NON sono esplorazione transitoria →
non si auto-correggono col budget. Meccanismo: ogni crossing mette il pedone in carreggiata
(project_to_road) → conflitto veicolo-pedone. Le rotte con crossing sono ~metà (hard 49/medium 45%) e
*portano* la ped-SR (hard 60 vs 43% non-crossing): un cap aggressivo le degraderebbe a respawn (16% SR).

**Modifica (evo/fix mirata).** `route_planner.py:261` `max_cross 8→3`: share crossing uguale medium/hard ma
hard usa più crossing *per rotta* (100m → più dead-end → cap morde di più) → cap=3 taglia la coda
ad alto-crossing (peggiori generatori di conflitto) risparmiando le tipiche rotte 1–2 crossing che reggono
la ped-SR. + telemetria `route_n_crossings` per record (`carla_multi_agent_env.py`: slots/init/emit +
`_setup_pedestrian_route`), measurement-only (pattern C0). Compile OK, git diff --check OK.

**Validazione (la lancia l'utente).** 300K hard-locked, A/B pulito vs `20260617_103301` (cambia solo max_cross):
`python -m carla_core.training.train_carla_mappo --mode curriculum --difficulty path --timesteps 300000 --seed 999 --lock-curriculum-level hard`
Gate: collision/offroad veicolo in calo verso il regime medium SENZA affondare la ped-SR hard. GO 3M solo se hard passa.

---

### [15-06-2026] PIANO — P1-Crosswalk (sostituisce il P1 branch-aware, falsificato dall'audit)

**Obiettivo**: convertire le route fallback degeneri in route sidewalk contigue reali cucendo gli
attraversamenti; distanze 30/60/100 m invariate.

**Modifiche** (≤150 parole):
- `route_planner.py::plan_pedestrian_route_by_distance`: al dead-end della catena greedy, trova il crosswalk
  più vicino entro `cross_radius` (~6 m) via `carla.Map.get_crosswalks()` (parsing poligoni + cache per-mappa);
  genera waypoint intermedi sull'attraversamento (passo `spacing`) fino al curb opposto; risolvi a sidewalk e
  prosegui. Cap `max_cross`; dedup celle anti-loop. `min_route_ratio` resta 0.5.
- `_setup_pedestrian_route` (env:1097): se neanche col crossing si raggiunge il floor, **respawn solo quel
  pedone** su RNG dedicato `SeedSequence([traffic_seed, reset_count, crc32(agent_id)])`; eliminato il fallback
  navmesh sconnesso e la catena legacy.
- Reward, obs, max_steps, codice veicolo: INVARIATI.

**Gate frozen A/B** (checkpoint r0610 `step_003000615`, seed appaiati, 4 scenari): meccanismo (ped fallback
≤1%, route_short ~0, ratio mediano ≥0.9); **veicoli per-livello ≥−2pp SR / ≤+2 s+t / ≤+1 coll·off** (binding);
ped SR ≥ baseline (sanity); integrità 6/6, zero NaN/inf. Il +5pp ped medium/hard è gate della fase finetune.

**Sequenza**: P0 (`set_pedestrians_seed` per-reset) → P1-Crosswalk → smoke test → frozen A/B → finetune ped-only
(~300-500K, `policies_to_train=["pedestrian_policy"]`, obs invariate ⇒ checkpoint-compatibile) → solo dopo
PASS, replica 3M from-scratch. Plan-only; run lanciate dall'utente.
**Spike crosswalk (audit `--crosswalks`, Town03, seed 999)**: 73 crosswalk, used_crossing 58.7%, mediana reach
contigua 40m→109m; fallback `min_route_ratio=0.5` easy 21→5.8%, medium 40→11.5%, hard 56→23.0%. Nessuna
terminazione offroad pedoni (offroad è vehicle-only); road penalty −0.3/step dominato da +50/waypoint. Trade-off
intrinseco ("via di mezzo"): route reali ⇒ ped attraversa ⇒ più interazione veicolo (hazard channel) → gestito
da frozen A/B + finetune ped-only.

---

### [19-05-2026] NEW PLAN — stack EVO (veicolo + curriculum + pedoni)

Context: run 20260519_001217 (3M) mostrava forgetting (SR medium veh 52.6→38.7%, ped 70.7→40.6% in fase hard),
hard force-unlocked a 0.552 del budget col veicolo a 46% SR, collision 6/18/25% easy/medium/hard. Obiettivo:
SR massimale veh+ped, minima degradazione multi-livello; solo evo/fix mirate, niente obs/architettura.

Candidati (1 per branch; ordine risolve gli overlap di file):
1. **Entropy** — schema-clean: `entropy_coeff_schedule_fraction [[0.0,0.03],[0.83,0.005]]` + preprocessor che
   lo espande in assoluto su `total_timesteps` (scala col budget; difendibile in tesi). [IMPLEMENTED]
2. **Ped-route** — `plan_pedestrian_route_by_distance` + `_setup_pedestrian_route`: rifiuta rotta <target×0.5
   (`pedestrian_route_min_ratio: 0.5`). [implementato]
3. **Ped-speed** — `_pedestrian_reward`: banda comfort 0.8–1.8 → 1.2–2.6 m/s.
4. **P1** — `curriculum_batch_manager.py`: rimuove esclusione di easy; floor rehearsal easy≥0.10/medium≥0.20 in fase hard.
5. **P2** — `_balanced_policy_success_details`: mean(SR) → min(SR); unlock gated sulla policy più debole.
6. **P3** — `curriculum_batch.yaml`: `force_unlock_global_share_cap 0.55 → 0.70`.
7. **R-norm v2** — `_vehicle_reward`: abbandonato design v1 (÷route_wp_count → collasso magnitudine). Design
   corretto: `length_factor = TARGET_ROUTE_WP / route_wp_count` (TARGET=30) su TUTTI gli shaping per-step;
   terminale −500 invariato. Condizionale a Block 4 (skip se P1+P2+P3+Ped-route appiattiscono il gradiente
   collision easy→hard).

Overlap file: P1+P2 (`curriculum_batch_manager.py`); R-norm/Ped-route/Ped-speed (`carla_multi_agent_env.py`) →
ordinati, mai paralleli. **Ordine**: Entropy→(Ped-route→Ped-speed)→(P1→P2→P3)→R-norm.
Gate per candidato vs baseline 20260519_001217: <gate_policy> per-livello + vf_explained_var≥0.8 per fase;
P1/P2/P3 su final-eval per-scenario (hard SR +≥2pp, easy/medium ≥−2pp no-forgetting, Town05 non peggiora);
Ped-route diagnostico (ped route_length_ratio<0.5 → ~0%). Sempre 6 record/ep, 0 NaN/inf.

**After-gate (architettura, proposta)**: non usare 20260519_001217 come verdetto finale (ped contaminati da
rotte sidewalk corte). Rifare baseline MLP 47D pulita (no Attention/PopArt/GNN), poi screening 300K nell'ordine
MLP→PopArt→Attention→Att+PopArt→GNN→GNN+PopArt, poi 3-seed validation, poi long run comparativa. Priorità:
vehicle hard SR (primario), stuck+timeout (secondario), collision/offroad canary, easy/medium no-forgetting.

---

### [15-05/18-05-2026] Piano chirurgico originale — vehicle policy (8 punti) + route-EVO

Baseline 211642 (easy-locked 300K): veh SR 20.1%, stuck+timeout 54.2%, collision 18.3%, offroad 7.4%; successi
lentissimi (15m a step ~862). Cause radice: (1) vf_clip=10 strozza il critic con returns O(10²–10³); (2)
orizzonte miope (20 Hz + γ=0.99 → ~5s vs episodi 50s); (3) gate reward `route_completion<0.3` spegne lo speed
shaping dopo il 30%; (4) bonus sterzo fluido incondizionato (+0.1/step da fermo); (5) violazione Markov
(no_wp_steps/loop non in obs); (6) route planner senza upper bound. 8 modifiche isolate, A/B vs 211642 (esiti → TRACK RECORD):

| ID | File · riga | Modifica |
|----|-------------|----------|
| H1+H1.1 | train_mappo.yaml:29 | vf_clip_param 10→1e6 (soglia |V−V_t| da 3.16 a 1000) + vf_loss_coeff 0.5→0.05 |
| H2 | train_mappo.yaml:23 | gamma 0.99→0.997 (orizzonte ~5s→~17s) |
| R1 | carla_multi_agent_env.py ~1760/1782/1788 | rimuove gate reward `route_completion<0.3` (guardie safe_to_push/alignment restano) |
| R2 | carla_multi_agent_env.py ~1803 | bonus sterzo fluido solo se speed_kmh>5 |
| P5 | route_planner.py ~184 | enforce upper bound `route_len ≤ 2.0× target` (contratto docstring) |
| O1 | carla_multi_agent_env.py:83 + centralized_critic.py:54 + _get_vehicle_obs | obs +2D: no_wp_norm + loop_flag |
| O2 | stesse costanti + _get_vehicle_obs | obs +1D: tempo residuo (44→47D; O1+O2 insieme, retrain from-scratch) |
| H3 | mappo_runtime.py + train_mappo.yaml | entropy_coeff_schedule decrescente |

Comparabilità: H/R/P5 checkpoint-compatibili col trunk; O1/O2 rompono i checkpoint (non confrontabili col 44D,
<do_not_infer>). Una modifica per run; dopo ogni decisione aggiornare CLAUDE.md/AGENTS.md/docx.

**Route-EVO + logging (18-05, applicato)**: multi-candidate A* (`vehicle_route_candidate_attempts=32`), contatori
candidati + latency, `route_under_target_flag`, `route_target_error_m`, propagati a TensorBoard + episodes.jsonl;
default `vehicle_route_min_ratio=0.5 / max_ratio=2.0`. Distanze curriculum ripristinate easy 30/medium 60/hard
100m (test 80m); fallback veicolo ancora legacy ~30m → su medium/hard accorcia (da sorvegliare via
route_fallback_rate). **Route-seed fix**: `hash(agent_id)` → `SeedSequence([traffic_seed, reset_count,
crc32(agent_id)])` (riproducibilità/route-pairing; bugfix, non candidato).


#########################################################  TRACK RECORD / VALUTAZIONI (recente → vecchio)  #########################################################

> Tabelle A/B numeriche complete: CLAUDE.md Candidate Registry + docs/EXPERIMENT_REGISTRY.md. Qui: verdetto + meccanismo + run ID.

**[26-05] Step5 3M `20260525_205912`** (EVO/curriculum-stack: V1+P1+P2+P3, difficulty=path, primo run a esercitare l'unlock).
Integrità 3318 ep × 6 OK, no NaN/inf, training healthy (vf_explained_var 0.967/0.964, num_env_steps 3.006M).
Veh cum SR 55.16% (SR_raw=corr, no false-complete), s+t 24.77%, coll 13.45%, off 6.61%, speed 19.83 km/h.
Ped cum SR_corr 40.24% (raw 62.13%; −21.89pp = 2179 falsi-completi, 93.5% da sidewalk_fallback). Per-livello veh
SR easy 67.76 / medium 55.79 / hard 33.18; ped SR_corr 65.30 / 33.72 / 14.31. Binding = hard (veh coll 22% / off
11%). Unlock: primo medium ep 870 (26%), primo hard ep 2415 (73%) → hard solo 20% del budget (vs 35%, sbloccato
tardi). Ped speed 1.77 m/s ∈ [1.5,2.2]. Q1→Q4 cum +3.2pp (no collapse; il declino per-quarter è task-shift, non
degrado). Su slice easy vs V1 (paired): SR +5pp, s+t −5pp, coll +0.1 → cumulativo extended healthy.

| Candidato | Run ID | Verdetto | Core (meccanismo / Δ chiave) |
|-----------|--------|----------|------------------------------|
| **V1** safe_to_push 0.75→0.85 | 20260525_162428 | **PROMOSSO (trunk)** | Gate 5/5 vs EvoEntropy: SR +2.28, s+t −2.97, coll −1.02, off −1.45, ped SR +0.22. Urgency/min_speed restano attivi sotto hazard∈[0.75,0.85) → rompe l'equilibrio difensivo. Q3→Q4 +5.1pp (unico in salita forte). Merge → curriculum-stack (acc152b). |
| **V3** ped band [1.2,2.6]→[1.5,2.2] | 20260525_125127 | RIGETTATO/revert | Gate2 FAIL (ped speed 2.641, +0.441 sopra). Band stretta + bonus asimmetrico → ped overshoot, veh defensive equilibrium, Q3→Q4 −4.1pp. Motiva V1. |
| **Bundle** Ped-route + Ped-speed | 20260525_091300 | gate parziale / retained | Gate1 PASS (ped SR +0.04), Gate2 FAIL (ped speed 2.281). Ripple veicolo (file condiviso): stuck +4.70, speed −2.27. Meccanismo: ped più veloci saturano hazard → safe_to_push=False → veh zero-speed optimum. sidewalk_fallback 7→20.7% (relabel-only). |
| **EvoEntropy** (step 1) | 20260520_133747 | PROMOSSO/validato | Schedule entropy fraction-based su trunk 47D/R1. Meccanismo confermato (entropy ↓ monotona, KL stabile). Baseline per Bundle/V1/V3. veh SR 60.48%, ped SR 85.67% (RAW, pre-d84fa3a). |
| **O1+O2** obs 44→47D | 20260518_152016 (poi 195947) | valutato/healthy/baseline 47D | vs 212109: SR −2.65, off +4.47 → non batte il trunk; ma critic sano (vf_explained_var 0.966). Diagnosi route: legacy_fallback 76.6% (A* fallisce spesso) → easy "15m" è mix A* corte + fallback ~28m. Non checkpoint-compat col 44D. |
| **P5** route-len upper bound | 20260517_212109 | tenuto (env correctness) | Gate 4/4 ma SR +49.57pp = **task-distribution, NON policy** (network/reward/obs identici; episodi 342→529, step medi 900→574, route% 50→87). Pre-fix accettava rotte >2× target non finibili. Tutte le run H/R erano su distribuzione contaminata. 212109 = nuova baseline post-bugfix. |
| **R2** smooth-steer gated su speed | 20260517_164707 | provvisorio/non promosso | A episodi pari: SR −2.15 (FAIL), s+t −2.05 (PASS reale), off +3.28 (FAIL). Riduce immobilità ma la mobilità extra → offroad, non completamenti. |
| **R1** rimozione gate rc<0.3 | 20260517_134652 | PROMOSSO (trunk) | Primo PASS 4/4 dopo D3/H1/H2/H3/R3: SR +4.69 (+45 completi assoluti, dalla coorte timeout), s+t −3.45, coll −1.46, off +0.22. Speed 11.5→13.0. (Caveat: interagisce con la lunghezza rotta → più esposto al re-check post-P5.) |
| **R3** collision penalty −50→−500 | 20260516_200545 | non promosso/ipotesi falsificata | Collision PIATTA (−0.23pp). La policy risponde su offroad (−3.32) ma non su collision → **evitare collisioni non è imparabile dalle obs 44D** (limite di percezione, non peso-reward). −500 ritenuto per decisione utente. |
| **H3** entropy schedule | 20260516_144007 | non promosso/meccanismo confermato | Entropy 4.78→3.25 (meccanismo OK) ma SR Δ solo-denominatore (216 completi in H2 e H3). Δ entro il rumore run-to-run. |
| **H2** gamma 0.99→0.997 | 20260515_211055 | non promosso/ipotesi falsificata | Collision +5.18pp: orizzonte lungo converte timeout→collision 1:1; SR piatta. Ritenuto per decisione utente. |
| **H1+H1.1** vf_clip 10→1e6 + vf_loss 0.5→0.05 | 20260515_175921 | non promosso/non revertato | Meccanismo: vf_explained_var ~0→0.87. Gate FAIL 3/4 (coll +3.28). Ritenuto: revertare ripristina un critic non-funzionale. |

**Quadro H1+H2+H3**: tre single-knob lato-ottimizzatore, tutti meccanismo-confermati, tutti gate-FAIL, SR veh
inchiodata ~21.5% (pre-bugfix). R3 chiude: il plateau è delimitato da percezione/reward-structure, non
dall'ottimizzatore. (Serie D: C2v2-A/D1/D2/D3/D2-Safety in CLAUDE.md registry.)


######################################################### Q&A (recente → vecchio )#########################################################

### [15-06] Audit sidewalk chains → P1 "branch-aware" falsificato
Audit offline (600 seed, Town03): modello fallback predetto 21.0/40.2/56.5% ≈ empirico r0610 21.1/38.6/52.5%
(modello fedele). `mean_max_outdeg=0.97 (<1)` → le sidewalk non si ramificano, `next()[0]` non scarta nulla;
`recover=0.0%` a ogni target → esplorare i rami è inutile; 91% dead-end, mediana catena contigua ~40m. Verdetto:
i target ped 60/100m **eccedono la geometria di Town03** → l'11% SR ped hard è in parte task geometricamente
impossibile, non solo policy. Opzioni reali: **A Crosswalk stitching** (ricostruisce route reali via
`get_crosswalks()`), C adaptive chain (target=lunghezza raggiungibile), B decouple target ped 15/30/60.
→ Spike conferma A (reach 40→109m). **Decisione utente: A** → PIANO P1-Crosswalk (sopra).

### [12-06] Verifica report (Claude F5) → conferma + 4 riallineamenti di protocollo
Tutti i claim del report riprodotti su episodes.jsonl (ped SR su fallback = 0%, fallback hard 52.48%,
controfattuale 81/51/23%, planner segue solo nexts[0], fallback navmesh sconnesso, lifecycle: terminati rimossi
ma non fermati). Precisazioni: (1) `sidewalk_fallback` copre DUE path (catena legacy ~25m + navmesh sconnesso);
(2) il "fallback SR=0%" è in parte strutturale (56% arriva in fondo ma demotato a route_short); (3) lifecycle
peggiore — i walker completati continuano "fantasma" → con P1 più completamenti → più ghost-walker → audit
ghost prerequisito al frozen A/B. Coupling misurato favorevole: ped fallback non danneggia i veicoli (su
medium/hard piatto; su easy correla ma confondibile). Riallineamenti: (1) gate frozen **asimmetrici** (vincolanti
solo meccanismo planner + non-degrado veicoli per livello + ped SR sanity; il +5pp ped è gate della fase
training); (2) **no scene-regeneration** (rejection sampling → bias) — respawn solo quel pedone con RNG dedicato;
(3) banda accettazione calibrata da audit offline; (4) P0 **per-reset** (`set_pedestrians_seed(seed+reset)`).
Direzione P0→P1→frozen A/B→finetune ped-only confermata come la più protettiva per i veicoli.

### [11-06] Valuta r0610 + "via di mezzo" ped senza ledere i veicoli
> "Mantenendo questa SR ottimale per i veicoli dobbiamo fixare la SR pedoni, trovando una via di mezzo che non leda i veicoli ma massimizzi entrambe."
r0610: veh SR 62.38%, ped SR 37.92%, ped sidewalk_fallback 35.73% (SR su fallback 0%, fallback hard 52.48%) →
problema dominante = generazione route. Controfattuale ottimistico (se fallback = route valide): ped easy/medium/
hard 64→81 / 31→51 / 11→23%, aggregato ~55%. Coupling via hazard channel (ped_ttc/ped_occ → safe_to_push) → fix
ped possono toccare veh, da misurare non assumere. **P1-Pareto** (un solo candidato comportamentale): mantieni
30/60/100m; esplora tutti i rami sidewalk (non solo nexts[0]); accetta solo catene contigue a target valido;
respawn solo quel pedone su RNG dedicato; elimina i punti navmesh sconnessi; non toccare reward/speed/obs/
waypoint-radius/successo. **P0 obbligatorio** (`set_pedestrians_seed` prima dello spawn). Protocollo anti-regressione:
frozen-checkpoint A/B su r0610, seed appaiati, breakdown per livello/policy, gate (ped fallback ≤1%, route_short
≤2%, ped SR medium/hard ≥+5pp, veh SR ≥−2 / s+t ≤+2 / coll·off ≤+1). Policy già separate → finetune ped-only
possibile congelando vehicle_policy (impedisce forgetting, non la variazione di veh SR da coupling). NON scartare
r0610, NON disaccoppiare l'architettura (il coupling è parte della research question MARL).

### [29-05] La SR veicolo inferiore (r0529 vs replicate) è fix-A o rumore?
**Prevalentemente rumore run-to-run.** Il gap esiste già nei primi 200 ep easy (nessuna decisione curriculum
ancora, stesso seed/codice/config): Δ veh SR −11.17pp su easy E −10.83 su medium (stessa magnitudo = stesso
noise floor, non regressione additiva). Lo stesso seed non è deterministico (cuDNN atomici FP, CUDA reductions,
RLlib worker async, CARLA TM threading, hash randomization, scheduling Windows) → due "stessi-seed" divergono su
1.5M step. **Noise floor osservato**: ~±10pp a 200/440 ep, ~±3-5pp a ~1500 ep, ~±2pp a 3000+ ep (le 300K
registry variano ~2pp su finestra full). Implicazione tesi: per claim affidabili servono **≥2-3 seed per
condizione** (curriculum vs mixed-batch) o confronto su Q3+Q4 della 3M (n grande, noise ~2pp).

### [26-05] A cosa serve episode_classification.py?
Modulo puro (no CARLA deps), `classify_termination_reason(...)` + `TERMINATION_REASONS` (alive, collision,
route_complete, route_short, offroad, stuck, timeout). Estratto da `step()` per testabilità (22 unit test, 0.002s)
e per il fix **Block-5.1**: demote `route_complete → route_short` se `optimal_length/target < 0.5` → SR ped non
più gonfiata da route fallback degeneri (nel 3M: 21.89% erano falsi-completi, 93.5% da sidewalk_fallback). I 3
aggregatori SR fanno `Counter.get("route_complete")` → route_short cade fuori automaticamente. Impatta solo run
future (Python non hot-reloada).

### [25-05] Serie V (5 Q&A connesse a VALUTAZIONE 20-05 punto 2)
1. **Patch V3 mostrata**: `_pedestrian_reward` band [1.2,2.6]→[1.5,2.2] (1 riga). Caveat: su hard 100m servono
   ≥2.0 m/s; se converge <1.8 allargare a [1.5,2.4].
2. **Recap piano**: apply V3 → 300K → eval → (PASS+stuck ok) step5 / (PASS+stuck high) V1 / (FAIL gate2) retune.
   V1+V2 NON in bundle (attribuzione). Step5 solo dopo promozione a trunk (altrimenti curriculum-stack eredita il ripple).
3. **Sequenza su EVO/new-main** (no branch dedicati, 1 commit/patch): V3→[PASS&stuck↓ trunk / PASS&stuck high
   +V1 / FAIL revert]→V1→[PASS trunk / FAIL +V2]. Gate α (solo ped SR+speed) vs β (+ side-check veh stuck ≤24%):
   decisione utente "valutiamo con entrambi".
4. **Anti-plateau**: il "vantaggio V3 sui veicoli" è illusorio (stuck = ritorno a EvoEntropy, SR Q4/speed
   peggiori dei 3). Cause plateau Q3→Q4: (1) defensive equilibrium (peds saturano hazard → safe_to_push spegne
   urgency/min_speed), (2) entropy collapse a 83%, (3) no_wp_steps penalty satura troppo presto. **Revert V3 →
   V1** attacca il meccanismo primario (gate safe_to_push). Opzioni V1 (safe_to_push 0.85) / V2 (no_wp cap
   1.0→2.0) / V4 (entropy 83→90%) / V5 (ped band asimmetrica −0.2 se >2.5). Eseguito: revert V3, apply V1.
5. **Cosa serve per PASS la 3M (step5)**: gate decomposti — **hard** (integrità 6/6; tutti 3 livelli raggiunti;
   budget share ±5pp; easy veh SR Q4 ≥75%; ped SR aggregata ≥70%; Q4 veh SR ≥ Q3−5pp) + **soft** (medium veh
   <50% → V4; hard veh <30% → V2; hard coll >15% / off >10% → retract V1; hard s+t >40% → V2; ped speed drift
   >0.3 → V5; ped SR hard <60%). Range attesi (orientativi): veh SR easy 78-85 / medium 55-70 / hard 35-55.
   Asimmetria del rischio: V1 validato solo su easy 30m → su hard 100m safe_to_push=0.85 è il punto più fragile.

### [20-05] Q&A varie
- **Testare i 7 candidati 1-by-1 o in blocco?** Si dividono per osservabilità: R-norm/Ped-speed/Entropy
  osservabili a 300K easy-locked (meccanismo per-step); P1/P2/P3 NO (si attivano post-unlock hard) → richiedono
  long run. Raccomandazione: 3 run (Entropy 300K; Ped-route+Ped-speed 300K; P1+P2+P3 in una 3M). Mai 7-in-blocco
  (zero attribuzione).
- **Patch veicolo (punto 2)?** Diagnosi: lo stuck è la stessa failure mode, più frequente (no_wp_steps 745→808).
  Meccanismo (L1911-1946): peds più rapidi → ped_ttc/ped_occ saturano → safe_to_push=False → urgency/min_speed
  off → veh zero-speed optimum; no_wp penalty cap troppo lento. Patch: **V1** (safe_to_push 0.75→0.85), **V2**
  (no_wp cap 1.0→2.0), V3 (ped band), V4 (accept-and-proceed). Tutte reward-shaping (no obs/arch).
- **Difficulty as-is 30/60/100 vs 15/30/60 con forbice?** Non cambiare in corsa. Prossima iterazione: 15/30/60
  punto fisso (B, salti 2×/2× puliti, easy=15m dove la 47D chiude). Run finale di tesi: forbice troncata [L,U]
  con moda al lower bound (diversità intra-stage → transfer Town05). Mantieni windowed-SR come gate.
- **Promotion mean o min?** Attualmente **media aritmetica** veh/ped (`_balanced_policy_success_details:422`),
  gate per-policy disattivato (`require_policy_success:false`). Con ped ~95% e veh ~50% la media ~72% non blocca.
  Per far vincolare il veicolo: `require_policy_success:true`, oppure mean→min (one-liner) [= P2], oppure media pesata.

### [18-05] Q&A varie
- **Vantaggio A* vs route statica/crescente?** A* per-episodio espone a una distribuzione di incroci/curve →
  competenza trasferibile (la route statica fa memorizzare 1 traiettoria → svuota la research question
  curriculum-vs-mixed e il test Town03→Town05). Difficoltà = asse pulito (più lunga = più incroci, non un layout
  scelto a mano); validità road-graph gratis; 3 veicoli RL hanno route distinte; gate sensati solo su
  distribuzione. I fix [0.5x,2.0x] + route-seed sono il modo corretto di pagare il costo di A*, non toppe.
- **Trigger unlock: windowed o cumulativa?** **Windowed**. L'unlock è un segnale di *controllo* (deve riflettere
  la competenza attuale); la cumulativa è un integratore in ritardo (pilota: 86% windowed vs 67% cumulativo) →
  tiene su easy oltre la mastery (budget sprecato) e su long run può non superare mai la soglia (degenera in
  schedule a timestep fissi). No conflitto con measurement_rule (cumulativa resta per il *reporting*). Rumore già
  mitigato da finestra 50 ep + min_budget_share + probation. Cambio narrow: far leggere a competence_unlocked la
  `window_success_rate` bilanciata.

### [undated] Operative
- **Dove sono i log TensorBoard?** In `C:\Users\kyros\ray_results\` (NON in carla_core/experiments), una dir
  per run (nome = hash, mappato per orario). `tensorboard --logdir C:\Users\kyros\ray_results`. Tag chiave:
  `…/vehicle_policy/learner_stats/vf_explained_var`, `…/grad_gnorm`, `custom_metrics/vehicle_policy/*_rate_mean`.
  Per il gate usare comunque episodes.jsonl (measurement_rules); progress.csv ha le 238 colonne.
- **Alzare max_steps 1000→1500 e troncare gli stuck in anticipo?** **NO a entrambi.** La coorte timeout NON è
  stuck (no_wp mediana 5, velocità terminale ~16 km/h ≈ successi): solo ~21% recuperabile (~+5pp), ed è un cambio
  di *metro di misura* (rompe la comparabilità di tutto il track record, SR sale a policy identica → pass di
  Potemkin). Il troncamento anticipato è **D3** (già rigettato: SR −2.90, s+t +7.07); il predicato proposto
  matcha 0 timeout, ri-etichetta solo episodi già persi → distorce collision/offroad. Rivalutare max_steps solo
  dopo P5 su rotte a lunghezza verificata.


#########################################################  BUGS / EVAL-PIPELINE  #########################################################

### [19-05] Bug routing pedonale — route troppo corte
Run 20260519_001217: pedoni con `route_length_ratio < 0.5` = 36.24% totale (easy 20.7 / medium 37.6 / **hard
53.6%**), `route_too_short_flag` 49.6%; veicoli 0/8784 (il lower-bound è rispettato solo per i veicoli). Causa:
`plan_pedestrian_route_by_distance` ritorna la catena anche se finisce prima del target, e `_setup_pedestrian_route`
(env:1117) accetta qualsiasi `len(wps)≥2` senza validare optimal_length/target. → Pedestrian hard
confounded/non-interpretabile. Patch: `pedestrian_route_min_ratio=0.5` (specchia il pattern veicolo). [→ poi Ped-route]

### [27-05] Eval-pipeline — 4 bug (eval-tooling only, nessun cambio training)
Prima eval di `20260525_205912` (deterministica): veh SR esattamente 33.33% su tutti i livelli, ped SR easy 1%
(vs 82% training) → firma di scena+azioni identiche ripetute. Tabella completa in CLAUDE.md "Eval Pipeline Bug
Fixes" + Candidate Registry (EvalBugfix, commit d0731ca + Patch B uncommitted).

| # | Bug | Fix |
|---|-----|-----|
| 1 | per-episode env-seed collassato a `seed_base` (evaluate_carla_mappo.py:248; reset_count letto solo se "maps") | `traffic.seed = seed_base + reset_count` |
| 2 | `deterministic_policy=true` → explore=False → mean della Gaussian (peds ~0.35 vs 2.0 m/s) | `false` su final_eval_job.json:143 + eval.yaml:7 + fallback mappo_runtime.py:340 |
| 3 | eval `episodes.jsonl` non scritto (`MAPPO_EPISODE_LOG` non propagato al child) | set env var nel child (`out_dir/eval/episodes.jsonl`) + clean-restart |
| 4 | torch/np seed correlato fra episodi (`exp_seed=seed_base` costante → noise stream identico, .debugging:292) | `exp_seed = seed_base + episode_idx` (Patch B) |

Caveat: protocollo training era già stocastico → l'eval stocastica è il confronto pulito. Residuo non-fixato:
collisione cross-scenario del seed (easy/medium/hard/test ep0 → seed 999; bounded perché route_distance_m
diverge dopo 1-2 step). Diagnosi confermata: stesso checkpoint, training stocastico cum veh 55% / ped 62%
(easy veh 67.76 / ped 82.42) = performance vera; l'eval deterministica NON è il riferimento per la tesi.
