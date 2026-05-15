# Analisi Critica Run: carla_mappo_20260514_211642

## 📋 Executive Summary & Context
**Stato della raccolta dati:** Completata (inclusa logica core per OBS/Reward/Iperparametri).
**Integrità:** Fonti chiave verificate; 1920 record (320 ep × 6 agenti).
**Vincoli rilevati:** * `--lock-curriculum-level`: **Easy** (rotte 15m).
* `final_evaluation_completed`: **False**.
* **Problema Critico:** La Success Rate (SR) aggregata dello 0.53 maschera un fallimento sistematico della policy veicoli.

---

## 📊 Analisi Metriche Disaggregate (Training)
Il confronto tra Veicoli e Pedoni evidenzia un forte squilibrio nelle prestazioni.

| Metrica (Training) | Veicoli (n=960) | Pedoni (n=960) |
| :--- | :--- | :--- |
| **Success Rate (route_complete)** | **20.1%** | **85.9%** |
| **Stuck** | 30.5% | 11.0% |
| **Timeout** | 23.7% | 3.0% |
| **Stuck + Timeout (Totale non-progressione)** | **54.2%** | **14.1%** |
| **Collision** | 18.3% | 0% |
| **Offroad** | 7.4% | N/A |
| **Route Completion (Media)** | 0.437 | 0.886 |
| **Velocità Media Finale** | 9.8 km/h | N/A |

> **Diagnosi Sintetica:** La policy veicolo ha adottato uno stile iper-conservativo. Il problema dominante è la **non-progressione** (54.2%), non la mancanza di sicurezza. Aumentare le penalità per collisione senza correggere lo shaping peggiorerà ulteriormente il blocco dei veicoli.

---

## Blocco 1: Analisi Osservazioni (OBS)
**Verdetto:** Geometricamente sufficienti, ma affette da violazione della **Proprietà di Markov**.

### Lacuna Rilevata
Esiste un evidente **State Aliasing**. La reward penalizza `no_wp_steps > 100` e `loop_penalty_active`, ma questi stati non sono mappati nell'input dell'agente (Obs 44D). L'agente non "vede" da quanto è fermo o quanto tempo rimane.

### Proposte di Refactoring Obs
* **O1 (Identità di Stato):** Estensione 44D -> 46D. Aggiunta `no_wp_steps` (normalizzato) e `loop-flag`.
* **O2 (Orizzonte Temporale):** Estensione -> 47D. Aggiunta `tempo_residuo` normalizzato: (max_steps - step_count) / max_steps.

---

## Blocco 2: Calibrazione Reward
**Verdetto:** Mis-calibrazione dei gate e bonus inconsistenti. **Non espandere, ri-calibrare.**

* **R1 - Rimozione Gate Progressione:** L'incentivo alla velocità si spegne a `route_completion >= 0.3`. Questo spiega il 23.7% di timeout (i veicoli "mollano" dopo il 30% della rotta).
* **R2 - Condizionamento Smoothness:** Il bonus per lo sterzo fluido (+0.1) è incondizionato. Un veicolo fermo accumula punti (~100/episodio), annullando di fatto le penalità di idle.
* **R3 - Conservazione:** Non aumentare la penalità collisione (attualmente 18.3%).

---

## Blocco 3: Iperparametri e Simulatore
**Verdetto:** Due criticità bloccanti a livello di ottimizzazione e orizzonte.

### H1: Clipping della Value Function (Priorità Max)
Il parametro `vf_clip_param=10.0` in `mappo_runtime.py` strozza il gradiente del Critic.
* **Problema:** Con returns O(10^2 - 10^3), l'errore quadratico supera quasi sempre sqrt(10).
* **Fix:** Alzare `vf_clip_param` a 10^4 o riscalare la reward (/100).

### H2: Disallineamento Orizzonte Temporale
Frequenza 20 Hz + gamma=0.99 -> Orizzonte effettivo di 5 secondi.
* **Problema:** Un episodio di successo dura ~43s (~862 step). L'agente è "miope".
* **Fix A:** Incrementare gamma a ~0.997.
* **Fix B:** Implementare `action-repeat` (frame-skipping) k=3 o k=4.

---

## Piano di Verifica (Gate Sequence)
Eseguire test isolati mantenendo il seed 999 e il protocollo 300k step.

1.  **Test H1:** `vf_clip_param` 10 -> 10000. (Verifica `vf_explained_var`).
2.  **Test H2:** gamma 0.99 -> 0.997.
3.  **Test R1/R2:** Modifiche allo shaping (Checkpoint compatibili).
4.  **Test Bugfix:** `route_planner.py` upper bound per lunghezze rotte.
5.  **Test O1/O2:** Cambio architettura (Variante separata).

---

## 📚 Riferimenti Scientifici
* **Time Limits:** Pardo et al. (ICML 2018) - Violazione Markov in env troncati.
* **MAPPO:** Yu et al. (2022) - Importanza della Value Normalization.
* **Reward Design:** Knox et al. (2023) - Problemi di mis-design nel driving RL.
* **Frame-skipping:** Mnih et al. (2015) / arXiv:2102.03718.




● Piano scritto. Riepilogo delle 8 modifiche chirurgiche (7 + 1 opzionale), tutte isolate e gate-driven:

  #: 1
  ID: H1
  File · riga: train_mappo.yaml:29
  Modifica: vf_clip_param 10 → 1e6 (sblocca l'apprendimento del critic)
  ────────────────────────────────────────
  #: 2
  ID: H2
  File · riga: train_mappo.yaml:23
  Modifica: gamma 0.99 → 0.997 (orizzonte 5 s → 17 s)
  ────────────────────────────────────────
  #: 3
  ID: R1
  File · riga: carla_multi_agent_env.py ~1760/1782/1788
  Modifica: rimuove il gate reward route_completion < 0.3
  ────────────────────────────────────────
  #: 4
  ID: R2
  File · riga: carla_multi_agent_env.py ~1803
  Modifica: bonus sterzo fluido solo se speed_kmh > 5
  ────────────────────────────────────────
  #: 5
  ID: bugfix
  File · riga: route_planner.py ~184
  Modifica: enforce upper bound 2.0x lunghezza rotta
  ────────────────────────────────────────
  #: 6
  ID: O1
  File · riga: carla_multi_agent_env.py:83 + centralized_critic.py:54 + _get_vehicle_obs
  Modifica: obs +2D: no_wp norm + loop flag
  ────────────────────────────────────────
  #: 7
  ID: O2
  File · riga: stesse costanti + _get_vehicle_obs
  Modifica: obs +1D: tempo residuo
  ────────────────────────────────────────
  #: 8 (opz.)
  ID: H3
  File · riga: mappo_runtime.py ~280 + train_mappo.yaml
  Modifica: schedule entropia decrescente

  Punti 1–5 sono checkpoint-compatibili e confrontabili col trunk; 6–7 cambiano la dimensione obs (44→47, da applicare insieme) e
  richiedono retrain da zero. Ogni punto è un A/B isolato da verificare col gate <gate_policy> contro la baseline 211642 (veicoli: SR
  20.1 %, stuck+timeout 54.2 %).

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Piano modifiche chirurgiche — Vehicle policy CARLA MAPPO

 Context

 L'analisi della run carla_mappo_20260514_211642 (easy-locked, 300k step, 320 ep, ricalcolata da episodes.jsonl) ha mostrato che la
 policy veicolo è in plateau:

 - SR veicolo 20.1 %, stuck 30.5 % + timeout 23.7 % = 54.2 %, collision 18.3 %, offroad 7.4 %.
 - Gli stuck hanno route_completion medio 0.15 e no_wp_steps medio 491 (mezzo episodio senza avanzare un waypoint).
 - Anche i successi sono lentissimi (rotta 15 m completata a step medio 862 ≈ 43 s).

 Cause radice individuate nella codebase:

 1. vf_clip_param=10 clampa la loss del value function (clamp((V−V_target)², 0, 10)): con returns O(10²–10³) il critic resta a
 gradiente ~nullo → vantaggi GAE degradati.
 2. Orizzonte miope: controllo a 20 Hz (fixed_delta_seconds=0.05, 1 tick/step) + γ=0.99 → orizzonte effettivo ~5 s contro episodi da 50
  s.
 3. Gate reward route_completion < 0.3: l'incentivo alla velocità si spegne dopo il 30 % di rotta (coorte timeout).
 4. Bonus sterzo fluido incondizionato: +0.1/step anche da fermo → cancella in parte la penalità idle.
 5. Violazione di Markov: la reward penalizza no_wp_steps e loop_penalty_active, ma nessuno dei due è nell'osservazione.
 6. Bug route planner: l'upper bound 2.0x della lunghezza rotta è nella docstring ma non nel codice → rotte "easy" di lunghezza non
 controllata.

 Obiettivo: 7 modifiche chirurgiche + 1 opzionale, ognuna applicabile e verificabile come A/B isolato (una modifica per run) contro la
 baseline 211642, senza toccare l'architettura MAPPO.

 Baseline di riferimento (run 211642, veicoli, da episodes.jsonl)

 ┌────────┬───────────────┬───────────┬─────────┐
 │   SR   │ stuck+timeout │ collision │ offroad │
 ├────────┼───────────────┼───────────┼─────────┤
 │ 20.1 % │ 54.2 %        │ 18.3 %    │ 7.4 %   │
 └────────┴───────────────┴───────────┴─────────┘

 Gate decisionale (CLAUDE.md <gate_policy>) — applicato a ogni A/B

 - Vehicle SR +≥2.0 pp
 - Vehicle stuck+timeout −≥2.0 pp
 - Collision e offroad non peggiori di +1.0 pp
 - Nessun NaN/inf in obs/reward/global_obs/metriche
 - Integrità episodio: 6 record agente/episodio

 Se un candidato fallisce il gate → revert solo di quel candidato, trunk C0+C1+D2 intatto.

 ---
 MODIFICHE (7 punti + 1 opzionale)

 Punto 1 — H1: alzare vf_clip_param (priorità massima)

 - File: carla_core/configs/train_mappo.yaml — riga 29
 - Esperimento: H1 · confrontabile col trunk: sì (nessun cambio obs/architettura, checkpoint-compatibile)

 Modifica:
 # PRIMA
   vf_clip_param: 10.0           # Value function clipping Reward v8
 # DOPO
   vf_clip_param: 1000000.0      # H1: clamp non vincolante — il critic deve poter apprendere returns O(10^2-10^3)

 Razionale: in RLlib (old API stack) vf_clip_param clampa la value-loss quadratica; con vf_clip=10 e returns nelle centinaia il
 gradiente al critic è ~zero. Un valore grande rende il clamp non vincolante. Sicurezza già presente: grad_clip=0.5 (norma globale) +
 vf_loss_coeff=0.5 + stop_on_nan=true.

 Verifica diagnostica aggiuntiva: controllare nei log RLlib/TensorBoard vf_explained_var (atteso: da ~0/negativo a >0.3) e vf_loss. Se
 compare instabilità/grad esplosivi → fallback: riscalare la reward ÷~100 (waypoint +100→+1) e tenere vf_clip_param ~50, ma è una
 modifica di reward (condizione separata).

 ---
 Punto 2 — H2: allungare l'orizzonte di sconto

 - File: carla_core/configs/train_mappo.yaml — riga 23
 - Esperimento: H2 · confrontabile col trunk: sì

 Modifica:
 # PRIMA
   gamma: 0.99
 # DOPO
   gamma: 0.997   # H2: orizzonte effettivo ~5s -> ~17s (1/(1-gamma) step x 0.05s)

 Razionale: a 20 Hz, γ=0.99 → orizzonte 1/(1−γ)=100 step = 5 s; un successo richiede ~862 step. γ=0.997 → ~333 step ≈ 17 s. Alternativa
  più profonda se γ da sola non basta (test successivo, non in questo punto): action-repeat k=3 in CarlaMultiAgentEnv.step()
 (carla_multi_agent_env.py:424), che porta il controllo a ~7 Hz e va accompagnato da max_steps ridotto.

 ---
 Punto 3 — R1: rimuovere il gate route_completion < 0.3

 - File: carla_core/envs/carla_multi_agent_env.py — funzione _vehicle_reward (righe ~1760, ~1782, ~1788)
 - Esperimento: R1 (reward shaping) · confrontabile col trunk: sì

 Modifica (3 micro-edit nella stessa funzione):

 (a) eliminare la riga ~1760 (variabile che diventa inutilizzata):
 # RIMUOVERE
             route_completion = self._route_completion(ad)

 (b) riga ~1782 — start/unblock shaping:
 # PRIMA
                 if safe_to_push and route_completion < 0.3 and alignment > 0.25:
 # DOPO
                 if safe_to_push and alignment > 0.25:

 (c) riga ~1788 — speed shaping (target_min_speed):
 # PRIMA
             if safe_to_push and route_completion < 0.3 and alignment > 0.25:
 # DOPO
             if safe_to_push and alignment > 0.25:

 Razionale: lo shaping di velocità/sblocco è inattivo oltre il 30 % di rotta; la coorte timeout (23.7 %, route_completion medio 0.55)
 sta esattamente lì senza incentivo a non rallentare. I gate safe_to_push (hazard<0.75) e alignment>0.25 restano e proteggono
 curve/ostacoli. Coefficienti invariati per un A/B a variabile singola; se R1 fa salire le collisioni oltre il gate, fallback:
 dimezzare i coefficienti su righe 1785-1786 e 1791-1793.

 ---
 Punto 4 — R2: condizionare il bonus di sterzo fluido alla velocità

 - File: carla_core/envs/carla_multi_agent_env.py — funzione _vehicle_reward (riga ~1803)
 - Esperimento: R2 (reward shaping) · confrontabile col trunk: sì

 Modifica:
 # PRIMA
         steer_delta = abs(ctrl.steer - ad.prev_steer)
         if steer_delta < 0.1:
             reward += 0.1   # smooth driving bonus
         elif steer_delta > 0.5:
             reward -= 0.3   # jerk penalty
 # DOPO
         steer_delta = abs(ctrl.steer - ad.prev_steer)
         if steer_delta < 0.1 and speed_kmh > 5.0:
             reward += 0.1   # smooth driving bonus (solo se il veicolo si muove davvero)
         elif steer_delta > 0.5:
             reward -= 0.3   # jerk penalty

 Razionale: il bonus +0.1/step è incondizionato → un veicolo fermo con sterzo costante incassa ~+100/episodio (= un waypoint),
 cancellando in parte la penalità idle −0.15. speed_kmh è già calcolato a inizio funzione (riga ~1722). La jerk-penalty resta invariata
  (la brusquezza è negativa a qualsiasi velocità).

 ---
 Punto 5 — Bugfix: enforce upper bound lunghezza rotta

 - File: carla_core/envs/route_planner.py — funzione plan_vehicle_route (righe ~184-187)
 - Esperimento: bugfix env · confrontabile col trunk: sì

 Modifica:
 # PRIMA
         # Validate route length
         route_len = _waypoints_length(wps)
         if route_len < target_distance_m * 0.5:
             logger.debug(
                 "Route too short: %.0fm vs target %.0fm", route_len, target_distance_m,
             )
             return None
 # DOPO
         # Validate route length: enforce [0.5x, 2.0x] of target (docstring contract)
         route_len = _waypoints_length(wps)
         if route_len < target_distance_m * 0.5 or route_len > target_distance_m * 2.0:
             logger.debug(
                 "Route length %.0fm outside [0.5x, 2.0x] of target %.0fm",
                 route_len, target_distance_m,
             )
             return None

 Razionale: la docstring promette validazione in [0.5x, 2.0x] ma il codice controlla solo il lower bound → rotte "easy 15 m" possono
 essere arbitrariamente lunghe (A* che gira l'isolato), gonfiando i timeout e decalibrando la difficoltà del curriculum. Rotte scartate
  ricadono sul fallback legacy wp.next().

 ---
 Punto 6 — O1: osservabilità dello stato "stuck" (Markov)

 - File 1: carla_core/envs/carla_multi_agent_env.py — riga 83 (costante)
 - File 2: carla_core/agents/centralized_critic.py — riga 54 (costante)
 - File 3: carla_core/envs/carla_multi_agent_env.py — funzione _get_vehicle_obs, prima del return a riga ~1233
 - Esperimento: O1 (cambio obs) · confrontabile col trunk: NO (rompe i checkpoint, vedi avvertenze)

 Modifica (a) — costante env, riga 83:
 # PRIMA
 VEHICLE_OBS_DIM = 44
 # DOPO
 VEHICLE_OBS_DIM = 46     # O1: +2 (no_wp_norm, loop_flag)   [47 se applicato anche O2]

 Modifica (b) — costante critic, centralized_critic.py riga 54:
 # PRIMA
 _VEHICLE_OBS_DIM = 44
 # DOPO
 _VEHICLE_OBS_DIM = 46    # deve combaciare con VEHICLE_OBS_DIM dell'env  [47 se anche O2]

 Modifica (c) — _get_vehicle_obs, inserire prima di return self._sanitize_obs(obs, ad.agent_id):
         obs[42] = veh_occ
         obs[43] = ped_occ

         # O1 — osservabilità stuck (coerenza Markov con i termini reward no_wp/loop)
         no_wp = max(self._step_count - ad.last_wp_advance_step, 0)
         obs[44] = min(no_wp / 300.0, 1.0)
         obs[45] = 1.0 if ad.loop_penalty_active else 0.0

         return self._sanitize_obs(obs, ad.agent_id)

 Razionale: la reward penalizza no_wp_steps>100 e loop_penalty_active ma l'obs non li contiene → stati identici in obs con reward
 diverse (state aliasing) proprio nella coorte stuck. obs è allocato np.zeros(VEHICLE_OBS_DIM), indici 0-43 già usati, 44/45 liberi;
 valori in [0,1] → nessuna distorsione da _sanitize_obs. global_obs_dim si auto-ricalcola (216→222) via
 compute_global_obs_dim_with_mask.

 ---
 Punto 7 — O2: osservazione del tempo residuo (time-aware)

 - File 1: carla_core/envs/carla_multi_agent_env.py — riga 83 (la stessa costante del Punto 6)
 - File 2: carla_core/agents/centralized_critic.py — riga 54 (la stessa costante del Punto 6)
 - File 3: carla_core/envs/carla_multi_agent_env.py — funzione _get_vehicle_obs, subito dopo le righe O1
 - Esperimento: O2 (cambio obs) · confrontabile col trunk: NO

 Modifica (a/b) — costanti a 47: se si applicano O1+O2 insieme, VEHICLE_OBS_DIM = 47 e _VEHICLE_OBS_DIM = 47 (non 46).

 Modifica (c) — _get_vehicle_obs, aggiungere dopo le righe O1, prima del return:
         obs[44] = min(no_wp / 300.0, 1.0)
         obs[45] = 1.0 if ad.loop_penalty_active else 0.0

         # O2 — time-aware observation (episodio troncato a orizzonte fisso)
         max_steps = max(int(self.cfg["episode"]["max_steps"]), 1)
         obs[46] = 1.0 - min(self._step_count / max_steps, 1.0)

         return self._sanitize_obs(obs, ad.agent_id)

 Razionale: episodio troncato a 1000 step ma l'agente non ha "orologio" → due stati con tempo residuo diverso sono indistinguibili
 (state aliasing sulla coorte timeout). Indice 46 libero. global_obs_dim si auto-ricalcola (→225 con O1+O2).

 ▎ Raccomandazione esecutiva O1/O2: applicare O1 e O2 insieme come unica variante 47D in un solo retrain. Ogni cambio di dimensione obs
 ▎  rompe già la compatibilità dei checkpoint ed è non confrontabile col trunk 44D (<do_not_infer>): farne due retrain separati spreca
 ▎ compute senza guadagno di comparabilità. Punti 6 e 7 sono distinti come punti di codice ma vanno trattati come un solo esperimento.

 ---
 Punto 8 (OPZIONALE) — H3: schedule di entropia decrescente

 - File 1: carla_core/training/mappo_runtime.py — blocco .training(), dopo riga 280
 - File 2: carla_core/configs/train_mappo.yaml — blocco optimization:
 - Esperimento: H3 · confrontabile col trunk: sì · da fare solo dopo H1/H2

 Modifica (a) — mappo_runtime.py, aggiungere una riga nel .training():
 # PRIMA
             entropy_coeff=opt.get("entropy_coeff", 0.01),
 # DOPO
             entropy_coeff=opt.get("entropy_coeff", 0.01),
             entropy_coeff_schedule=opt.get("entropy_coeff_schedule"),

 Modifica (b) — train_mappo.yaml, blocco optimization: (aggiungere sotto entropy_coeff):
   entropy_coeff: 0.03
   entropy_coeff_schedule: [[0, 0.03], [250000, 0.005]]   # H3: alta esplorazione iniziale -> consolidamento

 Razionale: entropy_coeff=0.03 costante è alto per controllo continuo 2D con log_std stato-dipendente → rumore d'azione persistente che
  litiga con il bonus di sterzo fluido. Uno schedule decrescente mantiene esplorazione presto e consolida tardi. Se la chiave yaml è
 assente, opt.get(...) ritorna None → comportamento invariato (retro-compatibile).
 Nota: gli endpoint dello schedule sono in timestep — per il run completo da 3M scalare a [[0, 0.03], [2500000, 0.005]].

 ---
 Verifica

 Static check (dopo ogni modifica)

 python -m compileall carla_core\envs\carla_multi_agent_env.py carla_core\agents\centralized_critic.py
 carla_core\training\mappo_runtime.py carla_core\envs\route_planner.py
 git diff --check

 Protocollo A/B (una modifica per run, isolata)

 Stesso setup della baseline 211642 — curriculum mode, difficulty path, easy-locked, 300k step, seed 999:
 python -m carla_core.training.train_carla_mappo --mode curriculum --difficulty path --timesteps 300000 --seed 999
 --lock-curriculum-level easy
 Poi ricalcolare le metriche veicolo da episodes.jsonl del nuovo run (disaggregare veicoli/pedoni) e confrontare con la baseline usando
  il gate decisionale sopra.

 Diagnostica specifica per esperimento

 - H1: vf_explained_var deve salire da ~0 verso >0.3; nessun NaN (gate mechanistic indipendente dalla SR).
 - H2: verificare che i successi avvengano a step_count più basso (guida meno esitante).
 - R1/R2: controllare che collision/offroad restino entro +1.0 pp.
 - Punto 5: la distribuzione della lunghezza rotta per livello deve restringersi.
 - O1/O2: retrain da zero (checkpoint non caricabili); confrontare solo come variante 47D, non col trunk 44D.

 Ordine consigliato

 1 (H1) → 2 (H2) → 3 (R1) → 4 (R2) → 5 (route-len) → 6+7 (O1+O2 insieme); 8 (H3) opzionale dopo H1/H2.
 Validare almeno H1 sul protocollo 300k prima di lanciare il run completo da 3M, per non cementare il bug del critic nella baseline di
 curriculum.

 Avvertenze di comparabilità

 - Una modifica per run. Non sommare candidati (<technical_constraints>, isolamento sperimentale).
 - Punti 1-5 sono checkpoint-compatibili e confrontabili col trunk C0+C1+D2.
 - Punti 6-7 (O1/O2) cambiano la dimensione obs (44→47): rompono i checkpoint, richiedono retrain da zero e non sono confrontabili
 direttamente col trunk (<do_not_infer>). Trattarli come variante separata.
 - Dopo ogni decisione (promosso/rifiutato): aggiornare current_known_state in CLAUDE.md, AGENTS.md e
 reports/carla_finetuning_maggio_2026.docx (<track_record_maintenance>).

 File critici toccati

 ┌──────────────────────────────────────────┬────────────┐
 │                   File                   │   Punti    │
 ├──────────────────────────────────────────┼────────────┤
 │ carla_core/configs/train_mappo.yaml      │ 1, 2, 8    │
 ├──────────────────────────────────────────┼────────────┤
 │ carla_core/envs/carla_multi_agent_env.py │ 3, 4, 6, 7 │
 ├──────────────────────────────────────────┼────────────┤
 │ carla_core/envs/route_planner.py         │ 5          │
 ├──────────────────────────────────────────┼────────────┤
 │ carla_core/agents/centralized_critic.py  │ 6, 7       │
 ├──────────────────────────────────────────┼────────────┤
 │ carla_core/training/mappo_runtime.py     │ 8          │
 └──────────────────────────────────────────┴────────────┘
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

● TensorBoard (TensorFlow è il framework; TensorBoard è la dashboard). E sì, è lo strumento giusto
  — anzi, per H1/H1.1 è di fatto l'unico: vf_explained_var, vf_loss, grad_gnorm, policy_loss sono learner-stats di RLlib che il
  results.json del progetto non salva (lì ci sono solo SR/reward/collision). Ho verificato la run 211642.

  Dove sono i log (≠ cartella del progetto)

  I log RLlib non stanno in carla_core/experiments/... — stanno in C:\Users\kyros\ray_results\, una dir per run. La run 211642 è:

  C:\Users\kyros\ray_results\PPO_CarlaMultiAgent-v0_2026-05-14_21-16-51trct60fp\
  (timestamp 21:16:51 ≈ run-name 211642). Dentro: events.out.tfevents.* (TensorBoard), progress.csv (238 colonne), result.json,
  params.json.

  Comandi

  Vista di tutte le run (TensorBoard auto-scopre le sottocartelle):
  tensorboard --logdir C:\Users\kyros\ray_results --port 6006
  Poi apri http://localhost:6006. Se manca: pip install tensorboard.

  Vista A/B con nomi leggibili (consigliata — baseline vs H1 vs H1.1):
  tensorboard --logdir_spec baseline:C:\Users\kyros\ray_results\PPO_CarlaMultiAgent-v0_2026-05-14_21-16-51trct60fp,H1:C:\Users\kyros\ray
  _results\<dir_run_H1>,H1.1:C:\Users\kyros\ray_results\<dir_run_H1.1>

  Identificare la dir di una nuova run (i nomi sono hash, non il run-name del progetto → si mappano per orario). Subito dopo aver
  lanciato un training:
  Get-ChildItem $env:USERPROFILE\ray_results -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 FullName

  Cosa guardare (nomi tag/colonna esatti, verificati dal progress.csv)

  Scopo: H1 — critico sbloccato?
  Tag (filtra in TB con vf_explained o vehicle_policy): info/learner/vehicle_policy/learner_stats/vf_explained_var → baseline
    ~0/negativo, H1 deve salire >0.3 · …/vf_loss
  ────────────────────────────────────────
  Scopo: H1.1 — starvation della policy?
  Tag (filtra in TB con vf_explained o vehicle_policy): …/grad_gnorm (esplode = gradiente value domina) · …/policy_loss (se si
    appiattisce mentre vf_loss scende = policy starved) · …/entropy · …/kl · …/total_loss
  ────────────────────────────────────────
  Scopo: Outcome live
  Tag (filtra in TB con vf_explained o vehicle_policy): custom_metrics/vehicle_policy/{success,stuck,timeout,collision}_rate_mean ·
    …/route_completion_mean

  Note importanti

  - Per la decisione del gate usa comunque episodes.jsonl ricalcolato da disco (regola di misura del progetto, <measurement_rules>): le custom_metrics in TB sono la curva live, episodes.jsonl è la verità aggregata e disaggregabile veicoli/pedoni.
  - Senza TB: progress.csv nella stessa dir contiene tutte le 238 colonne (stessi nomi della tabella sopra) → apribile con pandas se
  preferisci grafici tuoi.
  - Opzionale (fuori dal piano): il timeseries di results.json non include vf_explained_var — il dato c'è già nel result dict
  (result["info"]["learner"]…, vedi train_carla_mappo.py:284). Aggiungere una riga al timeseries.append (train_carla_mappo.py:1105) lo
  porterebbe nella pipeline del progetto, così compare_results_carla.py può usarlo senza TB. Dimmi se vuoi che lo aggiunga al piano come punto extra.