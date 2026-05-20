
#########################################################15-05-2026##############################################################

### PIANO CLAUDE CARLA

● Ho completato la raccolta dati (run + intera parte di carla_core rilevante per OBS/reward/iperparametri: carla_multi_agent_env.py,
  route_planner.py, centralized_critic.py, mappo_runtime.py, train_carla_mappo.py, configs). Le ricerche web sono andate a rate-limit ma
   le fonti chiave sono verificate. Ecco l'analisi critica completa.

  ---
  Analisi run carla_mappo_20260514_211642 + codebase

  Premessa critica sulla run

  Tre cose vanno dette subito, perché cambiano la lettura di tutto:

  1. La run è --lock-curriculum-level easy: curriculum_lock.enabled=true. Non è un esperimento di curriculum — è un run easy-only con
  rotte path 15 m. Non dice nulla su budget/sampling/curriculum (coerente con <do_not_infer>).
  2. final_evaluation_completed: false (final_eval_job.json → "launch manually"). evaluation: {} vuoto. Non esiste evidenza Town05.
  Tutto ciò che segue è basato su episodi di training (episodes.jsonl, 1920 record, 320 ep × 6 agenti, 0 duplicati, integrità OK).
  3. Lo "0.53 success_rate" in results.json è la SR congiunta veicoli+pedoni e maschera il problema. Disaggregato (episodes.jsonl,
  ricalcolato da disco):

  ┌───────────────────────────────┬─────────────────┬────────────────┐
  │      Metrica (training)       │ Veicoli (n=960) │ Pedoni (n=960) │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Success rate (route_complete) │ 20.1 %          │ 85.9 %         │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Stuck                         │ 30.5 %          │ 11.0 %         │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Timeout                       │ 23.7 %          │ 3.0 %          │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Stuck + timeout               │ 54.2 %          │ 14.1 %         │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Collision                     │ 18.3 %          │ 0 %            │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Offroad                       │ 7.4 %           │ —              │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Route completion (media)      │ 0.437           │ 0.886          │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ no_wp_steps (media)           │ 211             │ 115            │
  ├───────────────────────────────┼─────────────────┼────────────────┤
  │ Velocità a fine ep. (media)   │ 9.8 km/h        │ —              │
  └───────────────────────────────┴─────────────────┴────────────────┘

  Lettura empirica chiave (non ovvia):

  - Lo stuck ha route_completion medio 0.15 e no_wp_steps medio 491: in media metà episodio senza avanzare un solo waypoint. L'87 %
  degli stuck ha causa low_route_completion, il 62 % no_waypoint_advance, il 54 % loop_penalty.
  - Il timeout è diverso: route_completion 0.55, velocità 14 km/h — guida, ma non finisce in tempo.
  - Le collisioni avvengono tardi (step medio 933) e a bassa velocità (6 km/h): non sono schianti aggressivi, sono urti striscianti
  mentre il veicolo arranca nel traffico.
  - Anche i successi sono lentissimi: route_complete termina a step medio 862 (= 43 s) per una rotta target 15 m. Velocità media reale
  durante l'episodio ≈ 0.7 m/s (~2.5 km/h); i 15 km/h sono solo la velocità istantanea al traguardo.
  - La SR veicolo nella window finale (0.16) è ≤ della cumulativa (0.20): a 300k step la policy veicolo è in plateau ~16–20 %, non in
  salita. Non è solo "poco training".

  Diagnosi sintetica: la policy veicolo ha imparato uno stile iper-conservativo e quasi immobile. Il problema dominante non è la
  sicurezza (18 % collisioni) ma la non-progressione (54 % stuck+timeout). Qualsiasi modifica che aumenti la cautela (es. penalità
  collisione più alta) peggiora il target primario.

  ---
  1. Le OBS attuali sono sufficienti?

  Verdetto

  Geometricamente sì — l'obs 44D (con le feature route-aware di C1) copre bene il task di guida. Ma c'è una lacuna reale: una violazione
   della proprietà di Markov. Non servono nuovi "sensori"; servono 2–3 scalari che rendano osservabile lo stato che la reward già usa.

  Evidenza critica

  La _vehicle_reward (carla_multi_agent_env.py:1795-1796, 1810-1811) penalizza:
  - no_wp_steps > 100 → fino a −1.0/step
  - loop_penalty_active → −1.0/step

  Ma né no_wp_steps né il loop-flag sono nell'osservazione. L'obs 44D contiene cinematica, route preview, hazard, prev_steer — ma nessun
   segnale di "da quanto sono fermo" o "tempo residuo". Conseguenza: due stati con posizione/velocità/rotta identiche ma no_wp_steps=50
  vs 500 sono indistinguibili in obs e ricevono reward diverse → state aliasing → il critic è costretto a predire la media → vantaggi
  pessimi proprio nel regime stuck. È esattamente la coorte che fallisce.

  Lo stesso vale per il timeout: l'episodio tronca a 1000 step ma l'agente non ha alcun "orologio". Un veicolo a route_completion=0.55
  con 200 step residui dovrebbe accelerare; con 800 può cruise — stessa obs → aliasing.

  Proposte (variante separata, NON il trunk 44D)

  O1 — Aggiungere no_wp_steps normalizzato + loop-flag (obs 44D→46D)
  - Cosa: obs[44] = min(no_wp_steps/300, 1), obs[45] = float(loop_penalty_active).
  - Validazione ×3: (1) Empirica — gli stuck hanno no_wp medio 491 e 54 % loop, e l'obs li nasconde; (2) Codice — la reward referenzia
  quantità non osservate (:1795, :1810); (3) Letteratura — proprietà di Markov, lo stato deve essere funzione dell'osservazione (Sutton
  & Barto); Pardo et al. 2018 mostra che termini dipendenti da "tempo nascosto" causano state-aliasing e instabilità.

  O2 — Aggiungere tempo residuo normalizzato (obs +1, → 47D)
  - Cosa: obs[46] = (max_steps − step_count)/max_steps.
  - Validazione ×3: (1) Empirica — 23.7 % timeout, coorte distinta che "guida ma non finisce"; (2) Pardo et al. 2018, ICML — task a
  orizzonte fisso: la time-aware observation va inclusa per non violare Markov; (3) Pratica standard in env troncati a orizzonte fisso
  (Gymnasium TimeLimit).

  Cosa NON aggiungere

  - No lidar/depth/semantica/camera. L'obs è un vettore di stato dalle API CARLA — e questo è corretto: nota che il weather è fisso
  ClearNoon ma è irrilevante perché non c'è input visivo. Aggiungere feature percettive = solo rumore e costo.
  - No allargamenti delle slot attori (già 3 veicoli + 2 pedoni + 4 hazard TTC/occupancy): l'offroad è solo 7.4 %, non è il collo di
  bottiglia.

  Gate O1/O2

  - A/B isolato vs baseline 211642 (300k, easy-lock, seed 999). Gate progetto: veicolo SR +≥2.0 pp, stuck+timeout −≥2.0 pp,
  collision/offroad ≤+1.0 pp, no NaN, 6 record/ep.
  - ⚠️  Avvertenza forte: cambiare la dimensione obs (44→46/47) rompe la compatibilità dei checkpoint e, per <do_not_infer>, non è
  confrontabile direttamente col trunk 44D. Per questo O1/O2 vanno ultimi (vedi piano sotto), come variante esplicitamente etichettata.

  ---
  2. Le reward sono ben calibrate?

  Verdetto

  No, non del tutto — la reward v5/D2 ha 2 mis-calibrazioni concrete che alimentano direttamente stuck e timeout. Ma la risposta a "ne
  aggiungiamo altre?" è un netto NO: ci sono già ~8 termini veicolo; aggiungerne è rumore e superficie di reward-hacking. Si ri-calibra,
   non si espande.

  Evidenza critica

  Problema R1 — l'incentivo alla velocità si spegne a route_completion ≥ 0.3.
  In _vehicle_reward:1782-1793, sia lo start shaping sia lo speed shaping (target_min_speed=8.0) sono dietro la guardia route_completion
   < 0.3. Oltre il 30 % di rotta, l'unico segnale positivo al moto è +100/waypoint (sparso) e il termine distanza ×4. La coorte timeout
  vive esattamente lì: route_completion medio 0.55 → ha superato la guardia → nessun incentivo a non rallentare.

  Problema R2 — il bonus di sterzo fluido è incondizionato.
  _vehicle_reward:1802-1804: steer_delta < 0.1 → +0.1 ad ogni step. Un veicolo fermo che tiene lo sterzo costante incassa +0.1/step ≈
  +100/episodio (= un intero waypoint), che cancella in parte la penalità idle (−0.15/step). Per un'auto ferma-ma-stabile, prima che
  scattino le penalità no_wp/loop (100–500 step dopo), il netto è ≈ −0.05/step: il bacino "non fare nulla" è troppo poco profondo
  all'inizio dell'episodio.

  Proposte

  R1 — Estendere l'incentivo di passo a tutta la rotta (rimuovere/sostituire la guardia rc<0.3)
  - Cosa: sostituire il gate route_completion < 0.3 con un termine di "passo" attivo per tutta la rotta (es. piccolo bonus/penalità su
  speed_kmh vs target_min_speed, condizionato solo a safe_to_push e alignment, peso ridotto ~0.5× del valore attuale per non gonfiare la
   varianza).
  - Validazione ×3: (1) Empirica — coorte timeout rc=0.55, oltre la guardia, 23.7 %; (2) Codice — :1782, :1788 mostrano il gate
  esplicito; (3) Teoria — uno shaping di progresso dev'essere consistente lungo lo stato-spazio (potential-based shaping, Ng, Harada &
  Russell 1999): un incentivo che sparisce a metà task introduce un ottimo locale di "rallenta dopo il 30 %".

  R2 — Condizionare il bonus di sterzo fluido alla velocità
  - Cosa: pagare +0.1 di smoothness solo se speed_kmh > ~5 (oppure scalarlo per min(speed/5,1)). La jerk-penalty resta.
  - Validazione ×3: (1) Empirica — 264/960 veicoli (27.5 %) finiscono a <1 km/h: il bonus li premia mentre sono fermi; (2) Codice —
  :1802-1804, bonus incondizionato; +0.1×1000 ≈ +100 = un waypoint; (3) Letteratura — i termini di shaping additivi non vincolati creano
   ottimi sfruttabili: principio centrale della letteratura sul reward-hacking / mis-design della reward di guida (es. Knox et al.,
  Reward (Mis)design for Autonomous Driving).

  R3 — NON aggiungere termini e NON aumentare la penalità collisione
  - Validazione ×3: (1) Empirica — la collisione è il 18.3 %, non il fallimento dominante (54.2 % stuck+timeout); alzarla spinge verso
  l'ottimo "auto congelata" e peggiora il gate (stuck+timeout −2pp); (2) Codice — la reward ha già 7–8 termini veicolo; ogni nuovo
  termine è varianza extra su returns già O(10²–10³); (3) Letteratura — Knox et al. 2023 documenta che le reward di guida multi-termine
  fatte a mano sono error-prone e che "più termini" raramente aiuta.

  Nota trasversale (rimanda al Blocco 3)

  La scala della reward (+100/waypoint, returns O(10²–10³), reward_std ~12000) interagisce in modo grave con vf_clip_param=10. La
  soluzione "pulita" è riscalare tutti i termini (es. ÷100: waypoint +100→+1), ma è una modifica di reward → va testata come condizione
  separata e va bundlata con R1/R2 o trattata insieme a H1. Vedi H1.

  Gate R1/R2

  A/B isolato vs 211642 (1 modifica per volta), stesso protocollo, gate progetto. R1/R2 sono checkpoint-compatibili (non cambiano
  obs/architettura) → confrontabili col trunk.

  ---
  3. Iperparametri rete neurale e simulatore CARLA

  Verdetto

  Rete neurale (capacità): MLP 256×2 Tanh per attore (44→256→256→4) e critic (216→256→256→1) — adeguata, non è il collo di bottiglia,
  non toccarla (anche <technical_constraints> lo vieta). Iperparametri di ottimizzazione/simulatore: NO, due problemi seri.

### H1 — vf_clip_param=10 strozza l'apprendimento del critic (priorità massima)

  mappo_runtime.py:283 usa PPOConfig stock con vf_clip_param=10.0. In RLlib (old API stack, 2.10) vf_clip_param clampa la loss del value function: vf_loss = clamp((V − V_target)², 0, vf_clip_param). Quando |V − V_target| > √10 ≈ 3.16, la loss clampata è costante → gradiente ZERO al critic.

  Qui i returns sono O(10²–10³) (un successo veicolo ≈ +1200÷1500; reward_mean joint ≈ 22000). L'errore quadratico di valore è quasi sempre ≫ 10 → il critic resta a gradiente ~nullo per gran parte del training. Conseguenza: i vantaggi GAE collassano verso "returns grezzi" (stile REINFORCE, alta varianza). Combacia con l'evidenza: il sotto-task facile (pedoni) impara comunque all'86 %, quello difficile (veicoli) è limitato dalla varianza e va in plateau al 18 %.

  - Validazione ×3+: (1) Sorgente RLlib — clamp(vf_loss, 0, vf_clip_param) in ppo_torch_policy; (2) Doc RLlib — vf_clip_param "sensitive to the scale of the rewards … if your expected V is large, increase this"; (3) Issue ray-project/ray#8908 "Should remove vf_clip param from PPO" + nota che il VF-clipping è "a common cause of user problems"; (4) MAPPO (Yu et al. 2022): la value normalization
  "never hurts … critical to strong performance" — il problema che PopArt risolverebbe è esattamente questo.
  - Fix isolato (consigliato per primo): alzare vf_clip_param a un valore che non vincoli mai (es. 1e4–1e6), mantenendo grad_clip=0.5
  come rete di sicurezza, senza toccare obs/reward/architettura → pienamente confrontabile col trunk.
  - Fix "proprio" (alternativo): riscalare la reward ÷~100 (vedi nota R3) e tenere vf_clip ~50–100. Più pulito numericamente ma è una modifica di reward.
  - Gate diagnostico (mechanistic, indipendente dalla SR): misurare vf_explained_var (e vf_loss) nei log RLlib/TensorBoard. Atteso pre-fix: vf_explained_var ≈ 0 o negativo, vf_loss "incollata" vicino a 10. Post-fix: vf_explained_var deve salire (>0.3–0.5). Poi gate progetto sulla SR veicolo.

### H2 — Disallineamento di orizzonte: 20 Hz + γ=0.99 → 5 s (priorità alta)

  step() esegue un world.tick() per step; fixed_delta_seconds=0.05 → controllo a 20 Hz, nessun frame-skip. Con γ=0.99 l'orizzonte
  effettivo è 1/(1−γ)=100 step = 5 s. Ma un episodio dura 50 s e una rotta di successo richiede ~862 step (43 s): la reward di
  route-complete è scontata di 0.99^862 ≈ 2·10⁻⁴ — invisibile. L'agente ottimizza una finestra di 5 secondi → miope, esitante, lento.
  Spiega direttamente lo stile "creep".

  - Validazione ×3: (1) Aritmetica — orizzonte 1/(1−γ); empiricamente i successi sono a step ~862 ≫ 100; (2) Frame-skip è parametro
  standard e potente (Mnih et al. 2015 DQN usa skip-4; An Analysis of Frame-skipping in RL, arXiv 2102.03718); (3) Norma in CARLA RL —
  frequenza di decisione tipica ~10 Hz e frame_skip≈3 (rassegna CARLA-RL); il nostro 20 Hz senza skip è più fine del normale.
  - Fix (due opzioni, testare una per volta):
    - A — action-repeat k=3–4 in step() (ripeti l'azione, k tick): controllo 5–7 Hz, orizzonte a γ=0.99 sale a 15–20 s, e
  raddoppia/triplica il throughput. ⚠️  cambia la dinamica del MDP (uno step = 0.2 s) → condizione separata, episodio in step più corto.
    - B — alzare γ a ~0.997 (orizzonte ~17 s) o 0.999 (~50 s). Una riga, MDP invariato → più confrontabile. Più semplice come primo
  test.
  - Gate: A/B isolato, gate progetto. Per l'opzione A, ricontrollare max_steps (1000/k) per mantenere ~50 s di episodio.
  H3 — entropy_coeff=0.03 costante (priorità bassa, ablazione attenta)
  0.03 è alto per controllo continuo 2D con log_std stato-dipendente (MAPPO usa tipicamente ~0.01) → rumore d'azione persistente su throttle/sterzo, che litiga con il bonus di sterzo fluido. Ma serve esplorazione per uscire dal plateau stuck → non abbassarlo
  
  "secco": meglio uno schedule di entropia decrescente (alta presto per esplorare, bassa tardi per consolidare). Validazione: (1) norme PPO continuo; (2) conflitto empirico entropy↔smoothness; (3) Andrychowicz et al. 2021 What Matters in On-Policy RL. 
  
  Da fare solo dopo
  H1/H2.

  Iperparametri CARLA / curriculum — note

  - Densità traffico "easy" non è easy. levels.yaml: nel curriculum path usato dalla run, easy ha 15 veicoli + 30 pedoni NPC, identici a
   hard (varia solo la distanza rotta). Esistono già levels_traffic/levels_mixed con easy = 5/10 NPC, non usati da path. Per far
  imparare la guida prima, una vera tappa easy a bassa densità è già nel codice. (Rilevante per la tesi: questo path non testa il
  curriculum di traffico.)
  - Bug route_planner.py:182-190: la docstring promette validazione lunghezza rotta in [0.5×, 2.0×], ma il codice controlla solo il
  lower bound (route_len < target×0.5). Una rotta "easy 15 m" può quindi essere arbitrariamente lunga (A* che gira l'isolato → 100 m+).
  Questo gonfia i timeout e decalibra la difficoltà del curriculum. Fix a basso rischio: aggiungere il reject route_len > target×2.0.
  - Già esplorati (non riproporre): i commenti in train_mappo.yaml mostrano che num_sgd_iter 15→10 e sgd_minibatch 256→512 sono stati
  testati e revertiti ("NO_PASS"). hidden_size, lr, clip_param, kl_*, grad_clip sono ragionevoli — lasciare. Un lr_schedule decrescente
  è polish opzionale, non prioritario.
  - batch_mode=complete_episodes + batch 8000 ≈ 8 episodi/update → ~40 update totali a 300k step: pochissimi. Conferma il
  sotto-training, ma la window-SR veicolo in plateau dice che non è solo questione di step.

  ---
  Piano di verifica — scala di gate (ordine consigliato)

  Ogni esperimento = una modifica isolata, protocollo identico a 211642 (300k, --lock-curriculum-level easy, difficulty path, seed 999), confronto vs baseline 211642 (veicolo: SR 20.1 %, stuck+timeout 54.2 %, collision 18.3 %, offroad 7.4 %). Gate progetto: SR +≥2 pp, stuck+timeout −≥2 pp, collision/offroad ≤+1 pp, no NaN, 6 record/ep. Se fallisce → revert solo quel candidato.

  #: 1
  Candidato: H1 vf_clip_param 10→1e4
  Tipo: Hyperparam
  Confrontabile col trunk?: Sì (checkpoint-compat)
  Note: Più economico, più alto leverage, zero rischio comparabilità. Da fare per primo.
  ────────────────────────────────────────
  #: 2
  Candidato: H2 γ 0.99→0.997 (o action-repeat k=3)
  Tipo: Hyperparam / env
  Confrontabile col trunk?: Sì (γ) / parziale (skip)
  Note: Testare γ per primo (più semplice).
  ────────────────────────────────────────
  #: 3
  Candidato: R1 rimozione gate rc<0.3
  Tipo: Reward shaping
  Confrontabile col trunk?: Sì
  Note:
  ────────────────────────────────────────
  #: 4
  Candidato: R2 smoothness gated su velocità
  Tipo: Reward shaping
  Confrontabile col trunk?: Sì
  Note:
  ────────────────────────────────────────
  #: 5
  Candidato: route-length upper bound (route_planner)
  Tipo: Bugfix env
  Confrontabile col trunk?: Sì
  Note: Basso rischio.
  ────────────────────────────────────────
  #: 6
  Candidato: O1+O2 obs 44D→47D
  Tipo: Obs change
  Confrontabile col trunk?: No (rompe checkpoint, <do_not_infer>)
  Note: Variante separata, solo se 1–5 non chiudono il gap.
  ────────────────────────────────────────
  #: —
  Candidato: H3 entropy schedule
  Tipo: Hyperparam
  Confrontabile col trunk?: Sì
  Note: Solo dopo 1–2.

  Raccomandazione operativa: il recap dice "next action: lanciare il full path-curriculum run 3M". Lanciare 3M step con vf_clip_param=10
   significa bruciare ~10× il budget su un critic strozzato e cementare il bug nella baseline di curriculum. Validare almeno H1 (e
  idealmente H2) sul protocollo 300k prima del run completo da 3M.

  ---
  Avvertenze di comparabilità

  - Distinguere sempre le condizioni sperimentali: H1/H2/H3 = hyperparam; R1/R2 = reward shaping; O1/O2 = obs change (incompatibile coi
  checkpoint, non confrontabile col trunk 44D). Non sommare candidati in un unico run.
  - Nessun dato Town05 esiste per questa run: non inferire qualità di generalizzazione/eval finale da episodi di training.
  - La run 211642 non può fungere da baseline di curriculum (è easy-locked).

  ---
  Riferimenti (verificati questa sessione salvo dove indicato)

  - Pardo et al. 2018, Time Limits in Reinforcement Learning, ICML — proceedings.mlr.press/v80/pardo18a
  (https://proceedings.mlr.press/v80/pardo18a.html) · arXiv:1712.00378 (https://arxiv.org/abs/1712.00378)
  - Yu et al. 2022, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO), NeurIPS D&B — arXiv:2103.01955
  (https://arxiv.org/abs/2103.01955)
  - RLlib PPO vf_clip_param — Ray docs / ppo.py (https://docs.ray.io/en/latest/_modules/ray/rllib/algorithms/ppo/ppo.html) · issue
  ray-project/ray#8908 (https://github.com/ray-project/ray/issues/8908)
  - An Analysis of Frame-skipping in Reinforcement Learning — arXiv:2102.03718 (https://arxiv.org/pdf/2102.03718)
  - Da conoscenza consolidata (non ri-verificati via web questa sessione per rate-limit): Ng, Harada & Russell 1999, Policy Invariance
  Under Reward Transformations, ICML (potential-based shaping); Sutton & Barto, RL: An Introduction (proprietà di Markov); Mnih et al.
  2015, Human-level control…, Nature (frame-skip 4); Andrychowicz et al. 2021, What Matters for On-Policy RL, ICLR; Knox et al. 2023,
  Reward (Mis)design for Autonomous Driving, Artificial Intelligence (arXiv:2104.13906).

  ---
  Posso, se vuoi, implementare l'esperimento #1 (H1) come modifica isolata di una riga in train_mappo.yaml (vf_clip_param: 10.0 →
  10000.0) con verifica python -m compileall e annotazione del razionale, pronto per l'A/B 300k. Dimmi se procedo o se preferisci prima
  discutere l'ordine dei gate.

✻ Cooked for 7m 41s

※ recap: Goal: improve the CARLA MAPPO vehicle policy (currently 20% success, 54% stuck+timeout). I finished the critical analysis of 
  run 211642 and the codebase. Next action: implement experiment H1, raising vf_clip_param from 10 to 10000 for an isolated A/B test. (disable recaps in /config)
  
  
### IMPLEMENTATION STEPS

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
  
  #: 9 (opz)
  ID: R4
  Alzare −50 (es. −500/−2000) o ridurre lo shaping denso, per rendere la collisione non trascurabile sulla scala O(10^4). È reward shaping (condizione sperimentale separata).

  Punti 1–5 sono checkpoint-compatibili e confrontabili col trunk; 6–7 cambiano la dimensione obs (44→47, da applicare insieme) e
  richiedono retrain da zero. Ogni punto è un A/B isolato da verificare col gate <gate_policy> contro la baseline 211642 (veicoli: SR20.1 %, stuck+timeout 54.2 %).

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 
### Piano modifiche chirurgiche — Vehicle policy CARLA MAPPO

 Context

 L'analisi della run carla_mappo_20260514_211642 (easy-locked, 300k step, 320 ep, ricalcolata da episodes.jsonl) ha mostrato che la policy veicolo è in plateau:

 - SR veicolo 20.1 %, stuck 30.5 % + timeout 23.7 % = 54.2 %, collision 18.3 %, offroad 7.4 %.
 - Gli stuck hanno route_completion medio 0.15 e no_wp_steps medio 491 (mezzo episodio senza avanzare un waypoint).
 - Anche i successi sono lentissimi (rotta 15 m completata a step medio 862 ≈ 43 s).

 Cause radice individuate nella codebase:

 1. vf_clip_param=10 clampa la loss del value function (clamp((V−V_target)², 0, 10)): con returns O(10²–10³) il critic resta a
 gradiente ~nullo → vantaggi GAE degradati.
 2. Orizzonte miope: controllo a 20 Hz (fixed_delta_seconds=0.05, 1 tick/step) + γ=0.99 → orizzonte effettivo ~5 s contro episodi da 50s.
 3. Gate reward route_completion < 0.3: l'incentivo alla velocità si spegne dopo il 30 % di rotta (coorte timeout).
 4. Bonus sterzo fluido incondizionato: +0.1/step anche da fermo → cancella in parte la penalità idle.
 5. Violazione di Markov: la reward penalizza no_wp_steps e loop_penalty_active, ma nessuno dei due è nell'osservazione.
 6. Bug route planner: l'upper bound 2.0x della lunghezza rotta è nella docstring ma non nel codice → rotte "easy" di lunghezza non controllata.

 Obiettivo: 7 modifiche chirurgiche + 1 opzionale, ognuna applicabile e verificabile come A/B isolato (una modifica per run) contro la baseline 211642, senza toccare l'architettura MAPPO.

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


#########################################################18-05-2026##############################################################

### UPDATE PLAN (LAST OPEN POINTS)

 > Analizza: - Bug determinismo route-seed (hash(ad.agent_id)) — le run non sono route-paired tra processi. + Recap Open-points.

● Sì, la diagnosi è corretta — è un bug reale, e l'ho confermato in codice.

  hash() su str in CPython è salato per-processo (PYTHONHASHSEED randomizzato di default — verificato: non è pinnato da nessuna parte nel repo). Quindi hash(ad.agent_id) cambia ad ogni avvio di processo: il seed RNG della rotta non è riproducibile e, anche con seed=999 fisso, le due run di un A/B vedono rotte diverse → non route-paired. Sito unico (carla_multi_agent_env.py:960), solo veicoli — i pedoni non usano RNG per la rotta.

  Due precisazioni tecniche:

  - Nessuna complicazione worker-index. Il repo non cattura EnvContext/worker_index e i porti CARLA sono fissi (2000/8000) → setup single-env. Quindi il fix semplice è anche sufficiente: non rischia di azzerare la diversità di rotte cross-worker (non ci sono più worker).
  - C'è anche un bug secondario. Il seed è additivo (reset*1000 + hash%10000): gli intervalli si sovrappongono e i 3 veicoli a
  reset_count vicini possono collidere sullo stesso seed (→ stessa rotta). Conviene chiuderli insieme usando
  np.random.SeedSequence([seed, reset_count, agent_key]) con agent_key = hash stabile (zlib.crc32(agent_id)): è la via idiomatica NumPy, mixa correttamente, niente overlap.

  Classificazione: bugfix di correttezza/riproducibilità (stessa categoria del Punto 5), non un candidate di policy. Non cambia la distribuzione delle rotte, solo la riproducibilità → niente gate. Va però notato: cambia le sequenze di rotte, quindi una run post-fix non è bit-identica a 212109.

  Sequenza che propongo — e sì, mi torna:

  1. Docs — aggiorno l'esito Punto 5 + registro il route-seed bug.
  2. Route-seed fix — prima di O1+O2, così O1+O2 e soprattutto le run-tesi (curriculum vs mixed-batch) + le final-eval saranno
  route-paired. Ti mostro il diff esatto prima di applicarlo.
  3. O1+O2 — retrain-from-scratch 47D sulla base corretta.
  4. Final eval.

  Caveat su O1+O2: vs 212109 non sarà perfettamente paired (212109 è pre-fix), ma O1+O2 cambia la dimensione osservativa → è già un confronto coarse from-scratch, quindi accettabile. Un re-baseline 44D post-fix per il pairing perfetto costerebbe ~2h: lo sconsiglio.


 ---
 MODIFICHE (7 punti + 1 opzionale)

### Punto 1 — H1&H1.1 (DONE) : alzare vf_clip_param (priorità massima) + abbassare vf_loss_coeff

 - File: carla_core/configs/train_mappo.yaml — riga 29
 - Esperimento: H1 · confrontabile col trunk: sì (nessun cambio obs/architettura, checkpoint-compatibile)

 Modifica:
 # PRIMA
   vf_clip_param: 10.0           # Value function clipping Reward v8
   vf_loss_coeff: 0.5
 # DOPO
   vf_clip_param: 1000000.0      # H1: clamp non vincolante — il critic deve poter apprendere returns O(10^2-10^3)
   vf_loss_coeff: 0.05			 # H1.1 ribilancia la quota della value-loss nella norma globale.

> Chiarimento: per H1 a prima iterazione avevi scritto 1e4 adesso leggo 1e6 è corretto?                                                 

● Sì, 1e6 (1000000.0) nel piano è voluto, non un refuso. Nell'analisi iniziale avevo scritto un range — "un valore che non vincoli mai (es. 1e4–1e6)" — e nel piano chirurgico ho fissato l'estremo alto. Il motivo è tecnico e vale la pena chiarirlo.

  Perché 1e6 e non 1e4. vf_clip_param clampa l'errore di valore al quadrato: la soglia sull'errore assoluto è √(vf_clip_param).

  ┌───────────────┬───────────────────────────────────────────────┐
  │ vf_clip_param │ soglia |V − V_target| oltre cui gradiente = 0 │
  ├───────────────┼───────────────────────────────────────────────┤
  │ 10 (attuale)  │ 3.16                                          │
  ├───────────────┼───────────────────────────────────────────────┤
  │ 1e4           │ 100                                           │
  ├───────────────┼───────────────────────────────────────────────┤
  │ 1e6           │ 1000                                          │
  └───────────────┴───────────────────────────────────────────────┘

  I value target qui sono O(10²–10³). A inizio training il critico predice ~0 → |V − V_target| ~ centinaia. Con 1e4 (soglia 100) una grossa fetta dei sample è ancora clampata proprio nella fase iniziale → il fix è solo parziale. Con 1e6 (soglia 1000) il clamp diventa praticamente non vincolante. Quindi 1e6 è il valore che davvero sblocca il critico; 1e4 era il fondo, troppo conservativo, del range.

  Caveat onesto — vf_clip da solo è un compromesso. grad_clip=0.5 clippa la norma globale del gradiente (attore + critico insieme, anche se sono reti separate: clip_grad_norm_ scala tutto con lo stesso fattore). Con vf_clip alto la value-loss è grande → il suo gradiente domina la norma globale → l'update dell'attore viene scalato giù. In pratica 1e6 scambia "critico congelato" con "policy rallentata". Nessun valore singolo di vf_clip risolve entrambi: il problema vero è la scala della reward (returns O(10³) invece di O(1–10)).

  Cosa ne consegue operativamente:
  - H1 con 1e6 resta valido come diagnostico: vf_explained_var (da ~0 → atteso >0.3) ti dirà inequivocabilmente se il critico era il collo di bottiglia — informazione utile a prescindere.
  - Monitora anche il progresso della policy (entropy, miglioramento SR). Se vf_explained_var sale ma la SR veicolo non passa il gate → è la firma della policy starvation → passa al fallback già nel piano: riscalare la reward ÷~100 e tenere vf_clip ~50 (value target O(1), nessun gradiente domina). È il fix pulito.
  - Alternativa per restare un test puramente hyperparametro (2 righe, niente reward/architettura): vf_clip 1e6 + abbassare
  vf_loss_coeff 0.5 → ~0.05, che ribilancia la quota della value-loss nella norma globale.

  Vuoi che aggiorni il piano — es. aggiungo la nota vf_loss_coeff come co-leva al Punto 1, oppure cambio il valore? Per come è scritto ora (1e6, isolato, con fallback reward-rescale) è coerente e va bene come primo A/B.

 Razionale: in RLlib (old API stack) vf_clip_param clampa la value-loss quadratica; con vf_clip=10 e returns nelle centinaia il
 gradiente al critic è ~zero. Un valore grande rende il clamp non vincolante. Sicurezza già presente: grad_clip=0.5 (norma globale) + vf_loss_coeff=0.5 + stop_on_nan=true.

 Verifica diagnostica aggiuntiva: controllare nei log RLlib/TensorBoard vf_explained_var (atteso: da ~0/negativo a >0.3) e vf_loss. Se compare instabilità/grad esplosivi → fallback: riscalare la reward ÷~100 (waypoint +100→+1) e tenere vf_clip_param ~50, ma è una modifica di reward (condizione separata).

 ---
### Punto 2 — H2 (DONE) : allungare l'orizzonte di sconto

 - File: carla_core/configs/train_mappo.yaml — riga 23
 - Esperimento: H2 · confrontabile col trunk: sì

 Modifica:
 # PRIMA
   gamma: 0.99
 # DOPO
   gamma: 0.997   # H2: orizzonte effettivo ~5s -> ~17s (1/(1-gamma) step x 0.05s)

 Razionale: a 20 Hz, γ=0.99 → orizzonte 1/(1−γ)=100 step = 5 s; un successo richiede ~862 step. γ=0.997 → ~333 step ≈ 17 s. Alternativa più profonda se γ da sola non basta (test successivo, non in questo punto): action-repeat k=3 in CarlaMultiAgentEnv.step() (carla_multi_agent_env.py:424), che porta il controllo a ~7 Hz e va accompagnato da max_steps ridotto.


● H2 confermato. Per darti l'istruzione chirurgica esatta verifico lo stato attuale di train_mappo.yaml (deve ancora contenere i valori H1+H1.1, o vanno ri-applicati):

● train_mappo.yaml contiene ancora i valori H1+H1.1 (vf_clip_param: 1000000.0, vf_loss_coeff: 0.05) — non li hai revertiti. Quindi per H2 serve una sola modifica.

  Modifica chirurgica — H2

  File: carla_core/configs/train_mappo.yaml — riga 23

  # PRIMA
    gamma: 0.99
  # DOPO
    gamma: 0.997   # H2: orizzonte effettivo ~5s -> ~17s (1/(1-gamma) step x 0.05s)

  Nient'altro. vf_clip_param e vf_loss_coeff restano come sono (1e6 / 0.05): il razionale di H2 — propagare la penalità collisione −50 sull'avvicinamento — funziona solo con il critic operativo di H1. Revertirli renderebbe H2 inutile (critic a varianza 0%, l'orizzonte lungo non avrebbe nulla da propagare).

  Disegno sperimentale — A/B isolato

  La run H2 differirà da 175921 per un solo knob (gamma). Quindi:

  - Confronto di isolamento H2: H2-run vs 175921 (non vs 211642). È l'A/B a variabile singola che misura l'effetto di gamma.
  - Verifica finale di pacchetto: quando H2 chiude, il blocco H1+H1.1+H2 va anche confrontato col trunk reale 211642, perché è quella la
   decisione di promozione per la tesi.

  Gate H2 (vs baseline 175921: SR 21.61 · stuck+TO 52.19 · coll 21.61 · off 4.59)

  ┌────────────────────┬──────────────────┬────────────────────────────────────────────────────────────────┐
  │  Metrica veicolo   │   soglia gate    │                              nota                              │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────┤
  │ SR                 │ ≥ 23.61 % (+2.0) │ —                                                              │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────┤
  │ stuck+timeout      │ ≤ 50.19 % (−2.0) │ —                                                              │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────┤
  │ collision          │ ≤ 22.61 % (+1.0) │ è l'ipotesi: H2 deve farla scendere, non solo "non peggiorare" │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────┤
  │ offroad            │ ≤ 5.59 % (+1.0)  │ —                                                              │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────────────────────────┤
  │ NaN/Inf · 6 rec/ep │ 0 · ok           │ —                                                              │
  └────────────────────┴──────────────────┴────────────────────────────────────────────────────────────────┘

  Diagnostica da osservare

  - collision rate — il test vero di H2. Se non scende, l'orizzonte da solo non basta → il follow-up sarà ricalibrare la penalità −50 (troppo piccola sulla scala O(10⁴). 
  
	-> Alzare −50 (es. −500/−2000) o ridurre lo shaping denso, per rendere la collisione non trascurabile sulla scala O(10^4). È reward shaping (condizione sperimentale separata).
  
  - step_count dei successi — atteso più basso (guida meno esitante con orizzonte lungo).
  - vf_explained_var — deve restare alta (~≥0.8). Caveat: γ=0.997 gonfia i target di valore ~2-3× → vf_loss crescerà ben oltre i ~25k attuali e vf_clip=1e6 potrebbe ri-vincolare il clamp. Se vf_explained_var cala sensibilmente, servirà alzare vf_clip (→1e8) — ma è una modifica separata, non aggiungerla in questo run.

 ---
 
### Punto 3 — R1: rimuovere il gate route_completion < 0.3

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

 Razionale: lo shaping di velocità/sblocco è inattivo oltre il 30 % di rotta; la coorte timeout (23.7 %, route_completion medio 0.55) sta esattamente lì senza incentivo a non rallentare. I gate safe_to_push (hazard<0.75) e alignment>0.25 restano e proteggono curve/ostacoli. Coefficienti invariati per un A/B a variabile singola; se R1 fa salire le collisioni oltre il gate, fallback: dimezzare i coefficienti su righe 1785-1786 e 1791-1793.


### R1 applicato e verificato.

  R1 — fatto

  Diff in carla_core/envs/carla_multi_agent_env.py, _vehicle_reward(), 3 righe (Punto 3 del piano):

  - rimossa route_completion = self._route_completion(ad) (ora variabile inutilizzata — verificato: nessun altro uso nella funzione);
  - if safe_to_push and route_completion < 0.3 and alignment > 0.25: → if safe_to_push and alignment > 0.25: (start/unblock shaping);
  - stessa modifica sul secondo gate (speed shaping target_min_speed=8.0).

  Verifica: python -m compileall OK; git diff --check pulito; il diff è esattamente le 3 righe previste, nient'altro.

  Effetto atteso: lo shaping start/unblock e di velocità resta attivo per tutta la rotta, non solo nel primo 30%. Le guardie
  safe_to_push (hazard < 0.75) e alignment > 0.25 restano → curve e ostacoli ancora protetti. Coefficienti invariati: A/B a singolo knob.

  Come testarlo (non lo lancio io)

  Stesso protocollo della serie, A/B single-knob vs run 20260516_200545 (R3):

  python -m carla_core.training.train_carla_mappo --mode curriculum --difficulty path --timesteps 300000 --seed 999
  --lock-curriculum-level easy

  È un run da ~2h (R3 ne ha impiegate ~1.9h). Lancialo tu quando vuoi — se serve, col prefisso ! nel prompt così l'output entra in sessione.

  Gate R1 (veicoli, cumulativo da episodes.jsonl vs R3): SR +≥2.0 pp, stuck+timeout −≥2.0 pp, collision/offroad ≤+1.0 pp. Canary da tenere d'occhio: la collision — è il rischio "shaping di velocità su tutta la rotta → più contatti". Fallback già nel piano se la collision sfora il gate: dimezzare i coefficienti sulle righe di shaping (start_gain e 0.04/0.15).

 ---
 
### Punto 4 — R2: condizionare il bonus di sterzo fluido alla velocità

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
 cancellando in parte la penalità idle −0.15. speed_kmh è già calcolato a inizio funzione (riga ~1722). La jerk-penalty resta invariata (la brusquezza è negativa a qualsiasi velocità).

 ---
 
### Punto 5 — Bugfix: enforce upper bound lunghezza rotta

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

 Razionale: la docstring promette validazione in [0.5x, 2.0x] ma il codice controlla solo il lower bound → rotte "easy 15 m" possono essere arbitrariamente lunghe (A* che gira l'isolato), gonfiando i timeout e decalibrando la difficoltà del curriculum. Rotte scartate ricadono sul fallback legacy wp.next().


### Punto 5 applicato e verificato.

  Punto 5 — fatto

  carla_core/envs/route_planner.py, plan_vehicle_route:

  -        # Validate route length
  +        # Validate route length: enforce [0.5x, 2.0x] of target (docstring contract)
           route_len = _waypoints_length(wps)
  -        if route_len < target_distance_m * 0.5:
  +        if route_len < target_distance_m * 0.5 or route_len > target_distance_m * 2.0:
               logger.debug(
  -                "Route too short: %.0fm vs target %.0fm", route_len, target_distance_m,
  +                "Route length %.0fm outside [0.5x, 2.0x] of target %.0fm",
  +                route_len, target_distance_m,
               )
               return None

  Verifica: compileall OK, git diff --check pulito; il diff è esattamente la modifica del Punto 3... pardon, del Punto 5 (validazione + messaggio di log).

  Effetto: una rotta veicolo A* più lunga di 2.0× il target ora viene scartata → fallback alla catena legacy wp.next(). La docstring prometteva [0.5x, 2.0x] ma il codice controllava solo il lower bound: ora il contratto è onorato. Una rotta "easy 15m" non può più essere arbitrariamente lunga.

  Come testarlo

  A/B single-knob vs la run R2 20260517_164707 (run più recente / base attuale):

  Gate (veicoli, vs R2): gate progetto standard. Diagnostica specifica del Punto 5: la distribuzione della lunghezza-rotta per livello deve restringersi (è il vero segnale del bugfix).

  Nota onesta sull'interpretazione. Punto 5 è un bugfix di calibrazione d'ambiente, non una leva di policy: accorcia/limita le rotte "easy", quindi un eventuale +SR riflette in parte "il task è diventato davvero 15m" e non "la policy è migliorata" (<do_not_infer>: distinguere misura da policy). Va valutato come "il bounding delle rotte cambia gli esiti / de-confonde i timeout", non come miglioramento di policy. Caveat di base: la run starebbe sopra R1+R2 (R2 non promosso) — l'A/B resta valido single-knob (cambia solo route_planner).

  (distinguere misura da policy). Va valutato come "il bounding delle rotte cambia gli esiti / de-confonde i timeout", non come
  miglioramento di policy. Caveat di base: la run starebbe sopra R1+R2 (R2 non promosso) — l'A/B resta valido single-knob (cambia solo route_planner).


 ---
 
### Punto 6+7 — O1: osservabilità dello stato "stuck" (Markov)

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
 diverse (state aliasing) proprio nella coorte stuck. obs è allocato np.zeros(VEHICLE_OBS_DIM), indici 0-43 già usati, 44/45 liberi; valori in [0,1] → nessuna distorsione da _sanitize_obs. global_obs_dim si auto-ricalcola (216→222) via
 compute_global_obs_dim_with_mask.

 ---
 
### Punto 6+7 — O2: osservazione del tempo residuo (time-aware)

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

 > Raccomandazione esecutiva O1/O2: applicare O1 e O2 insieme come unica variante 47D in un solo retrain. Ogni cambio di dimensione obs
 > rompe già la compatibilità dei checkpoint ed è non confrontabile col trunk 44D (<do_not_infer>): farne due retrain separati spreca
 > compute senza guadagno di comparabilità. Punti 6 e 7 sono distinti come punti di codice ma vanno trattati come un solo esperimento.


 ---
 
### Punto 8 (DONE) — H3: schedule di entropia decrescente

 - File 1: carla_core/training/mappo_runtime.py — blocco .training(), dopo riga 280
 - File 2: carla_core/configs/train_mappo.yaml — blocco optimization:
 - Esperimento: H3 · confrontabile col trunk: sì · da fare solo dopo H1/H2

 Modifica (a) — mappo_runtime.py, aggiungere una riga nel .training():
 # PRIMA
             entropy_coeff=opt.get("entropy_coeff", 0.01),
 # DOPO
             entropy_coeff=opt.get("entropy_coeff", 0.01),
             entropy_coeff_schedule=opt.get("entropy_coeff_schedule"),

 Modifica (b) — train_mappo.yaml, blocco optimization: (aggiungere sotto entropy_coeff): entropy_coeff: 0.03
   entropy_coeff_schedule: [[0, 0.03], [250000, 0.005]]   # H3: alta esplorazione iniziale -> consolidamento

 Razionale: entropy_coeff=0.03 costante è alto per controllo continuo 2D con log_std stato-dipendente → rumore d'azione persistente che litiga con il bonus di sterzo fluido. Uno schedule decrescente mantiene esplorazione presto e consolida tardi. Se la chiave yaml è assente, opt.get(...) ritorna None → comportamento invariato (retro-compatibile).
 Nota: gli endpoint dello schedule sono in timestep — per il run completo da 3M scalare a [[0, 0.03], [2500000, 0.005]].


### VALUTAZIONE RUN H1+H1.1+H2+H3

● Valutazione run carla_mappo_20260516_144007 (H3 — schedule entropy_coeff)

  Verdetto: H3 NON passa il gate veicoli (FAIL 3/4). Meccanismo confermato, ma effetto sull'outcome nullo entro il rumore run-to-run (NO REVERT)

  Confermato che è un A/B single-knob valido: run_config.json di H3 vs H2 differisce solo per entropy_coeff_schedule 
  [[0,0.03],[250000,0.005]] (gamma 0.997, vf_clip 1e6, vf_loss_coeff 0.05, seed 999, easy-lock, 300k identici).

  Gate veicoli — cumulativo da episodes.jsonl (H3 vs base H2 211055)

  ┌──────────────────────────────┬─────────────┬─────────────┬──────────┬────────┬─────────┐
  │ Metrica veicoli (cumulativa) │ H2 (211055) │ H3 (144007) │    Δ     │ Soglia │  Esito  │
  ├──────────────────────────────┼─────────────┼─────────────┼──────────┼────────┼─────────┤
  │ Success rate                 │ 21.75%      │ 21.43%      │ −0.32 pp │ ≥ +2.0 │ ❌ FAIL │
  ├──────────────────────────────┼─────────────┼─────────────┼──────────┼────────┼─────────┤
  │ Stuck+timeout                │ 44.71%      │ 44.35%      │ −0.37 pp │ ≤ −2.0 │ ❌ FAIL │
  ├──────────────────────────────┼─────────────┼─────────────┼──────────┼────────┼─────────┤
  │ Collision                    │ 26.79%      │ 25.79%      │ −0.99 pp │ ≤ +1.0 │ ✅ PASS │
  ├──────────────────────────────┼─────────────┼─────────────┼──────────┼────────┼─────────┤
  │ Offroad                      │ 6.75%       │ 8.43%       │ +1.69 pp │ ≤ +1.0 │ ❌ FAIL │
  └──────────────────────────────┴─────────────┴─────────────┴──────────┴────────┴─────────┘

  - Integrità: 336 episodi × 6 record = 2016, 0 duplicati, 0 episodi malformati → PASS.
  - NaN/inf: 0 su tutti i campi numerici → PASS.
  - Pedoni (riportati separatamente): SR 87.20% (H2 84.69%), collision 0.10% — forti e stabili, nessun effetto negativo.

  Meccanismo: confermato

  L'entropia veicolo a fine training (da last_result.json, iter 36): H2 = 4.78 → H3 = 3.25 con entropy_coeff correttamente sceso a 0.005. Lo schedule ha fatto esattamente ciò per cui è stato progettato: ha soppresso il blow-up tardivo di entropia (H1→5.43, H2→4.78, H3→3.25). vf_explained_var resta sano (0.92). Nessun danno al critic.

  Perché il meccanismo non muove l'outcome

  1. Successi veicolo identici in valore assoluto: H2 = 216, H3 = 216 route-complete. La SR scende di 0.32 pp solo perché H3 ha girato 5 episodi in più (denominatore). Lo schedule non ha prodotto neanche un successo in più.
  2. Le Δ sono dentro il rumore. Nei chunk 1–3 (timestep < 250k, dove H2 e H3 hanno config identica) la SR per chunk oscilla già di ±2–8 pp e la collision di ±7 pp tra le due run, per pura non-determinismo CARLA/RLlib. Le Δ cumulative del gate (−0.32, −0.37, −0.99, +1.69) non superano questo pavimento di rumore. Conferma indiretta: la velocità veicolo H3 = 10.78 km/h ≈ H1 (10.08), non H2 (13.81) — eppure H3 ha lo stesso gamma 0.997 di H2; quindi il presunto "gamma alza la velocità" attribuito a H2 era anch'esso in larga parte rumore.
  3. Decadimento tardivo della SR (chunk 4→6, ipotesi centrale di H3): H3 32.74→28.57→22.62, H2 32.73→28.48→19.64. Lo schedule attenua leggermente il crollo finale (+2.98 pp sul chunk 6), ma il picco (chunk 4 ~32.7%) è invariato e nel chunk 6 la collision H3 è peggiore (33.93 vs 30.36). Effetto reale ma piccolo e compensato.
  4. Spostamento di composizione: timeout −5.81 pp ma stuck +5.45 pp → entropia più bassa = policy più "decisa" che si incastra (stuck) invece di vagare fino al timeout, e quando sbaglia commit va offroad (+1.69) invece di restare passiva. S+T netto invariato.

  Quadro convergente H1 + H2 + H3

  Tre cambi single-knob consecutivi sul lato ottimizzatore — critic fix (H1), orizzonte (H2), entropia (H3) — tutti meccanicamente confermati, tutti falliti al gate veicoli ma non revertati (migliorano le metriche), con SR veicolo inchiodata a ~21.5% (successi assoluti 212/216/216). Evidenza ormai solida: il tetto della vehicle policy non è fissato dai knob ottimizzatore/critic/orizzonte/entropia. Il binding constraint resta la collision (25.79%), governata dalla struttura di reward — che è esattamente ciò che H1/H2/H3 non hanno toccato.

  Final eval

  Nessun file per 20260516_144007 in carla_core/results/eval/ (ultimi: 20260511). Coerente con H1/H2: il gate degli A/B single-knob
  H-series usa gli episodi di training cumulativi, non la final eval. Non è un blocco per la decisione.

  Raccomandazione

  1. Non promuovere H3 (registro: not promoted, gate FAIL 3/4 — di fatto risultato nullo entro il rumore).
  2. Prossimo step → R3 (penalità collisione veicolo −50 → −500 in carla_multi_agent_env.py:1748): è l'unico candidato che attacca il binding constraint dimostrato. La serie H ha esaurito il lato ottimizzatore.

### VALUTAZIONE R3

● Valutazione run R3 carla_mappo_20260516_200545

  Verdetto: R3 NON passa il gate adattato (FAIL). Risultato chiave: la penalità collisione 10× NON ha ridotto le collisioni — il
  presupposto di R3 è falsificato.

  Run completa e integra: 326 episodi × 6 = 1956 record, 0 duplicati, 0 malformati, 0 NaN/inf.

  Gate adattato R3 — veicoli, cumulativo da episodes.jsonl (R3 vs base H3 144007)

  ┌────────────────────────┬───────────┬───────────┬──────────┬────────┬─────────┐
  │                        │ H3 144007 │ R3 200545 │    Δ     │ Soglia │  Esito  │
  ├────────────────────────┼───────────┼───────────┼──────────┼────────┼─────────┤
  │ SR (primary)           │ 21.43%    │ 22.70%    │ +1.27 pp │ ≥ +2.0 │ ❌ FAIL │
  ├────────────────────────┼───────────┼───────────┼──────────┼────────┼─────────┤
  │ Collision (primary)    │ 25.79%    │ 25.56%    │ −0.23 pp │ ≤ −3.0 │ ❌ FAIL │
  ├────────────────────────┼───────────┼───────────┼──────────┼────────┼─────────┤
  │ Stuck+timeout (canary) │ 44.35%    │ 46.63%    │ +2.28 pp │ ≤ +1.0 │ ❌ FAIL │
  ├────────────────────────┼───────────┼───────────┼──────────┼────────┼─────────┤
  │ Offroad (canary)       │ 8.43%     │ 5.11%     │ −3.32 pp │ ≤ +1.0 │ ✅ PASS │
  └────────────────────────┴───────────┴───────────┴──────────┴────────┴─────────┘

  OVERALL: FAIL. Pedoni separati: SR 89.06% (H3 87.20), collision 0.20% — forti e stabili.

  Cosa ha fatto davvero R3

  La penalità −500 era attiva e ha rimodellato la policy — ma non sull'asse collisione. Spostamento delle terminazioni veicolo: stuck −7.17 pp, offroad −3.32 pp, timeout +9.45 pp, collision piatto, SR +1.27. Cioè: R3 ha convertito episodi stuck e offroad in timeout (+ qualche route_complete). route% 0.44→0.50, no_wp 290→248, entropy veicolo 3.25→2.58 (più "decisa"), vf_explained_var 0.92→0.88 (i picchi −500 aggiungono varianza al critic).

  Perché −500 non ha tagliato le collisioni:
  - La policy risponde alla pressione di reward dove può (offroad −3.32) ma non sulla collisione. Una policy che migliora l'offroad sotto pressione di reward ma non la collisione segnala che evitare le collisioni non è apprendibile dall'osservazione 44D attuale — è un limite di percezione, non di peso del reward.
  - In aggregato: costo collisione atteso = 25.6% × 500 ≈ 128 reward/episodio; bonus di route incassato ≈ route% 0.50 × ~7-8 wp × 100 ≈ 350-400. Il reward di route resta ~3× il costo atteso della collisione → guidare verso i waypoint resta il gradiente dominante. (Il mio calcolo EV pre-run, per-manovra, sottostimava il reward di route accumulato sull'intero episodio: lo riconosco.)
  - Salire ancora (−1500+) convertirebbe — per la lezione D3/D2-Safety — le collisioni in stuck/timeout, non in route_complete. Lo shaping della penalità collisione è un vicolo cieco a qualunque magnitudo.

  Conclusione strategica

  R3 falsifica "la collisione è tunabile via magnitudo della penalità". Con H1/H2/H3 (ottimizzatore esaurito) e la serie D, il plateau della vehicle policy (~21-23% SR) è ora delimitato dalla percezione/osservazione, non da ottimizzatore né pesi di reward. Da D2 in poi: 5 candidati consecutivi (D3, H1, H2, H3, R3) falliti al gate.

  Raccomandazione: R3 non promosso; consiglio di revertare −500→−50 (lo scopo di R3 è falsificato; portare nella tesi un termine di reward 10× non giustificabile è indifendibile; i guadagni marginali sono nel rumore). Decido con te qui sotto — R3 ha comunque l'SR più alta della serie e ha ridotto l'offroad, quindi è una scelta reale.

### VALUTAZIONE R1 

● Valutazione R1 — run carla_mappo_20260517_134652

  Verdetto: R1 PASSA il gate veicoli — 4/4 + integrità. È il primo PASS dopo D3, H1, H2, H3, R3.

  Integrità. 1950 record, 325 episodi × 6, 0 duplicati, 0 NaN/inf. seed=999, ~302k step, easy-lock → A/B single-knob pulito vs R3
  (20260516_200545). R3 ricalcolato da disco in parallelo (numeri identici alla valutazione precedente).

  Gate veicoli (cumulativo da episodes.jsonl, gate progetto standard — R1 mira a stuck+timeout, quindi vale il gate normale, non
  adattato):

  ┌─────────────────┬────────┬────────┬──────────┬────────┬─────────┐
  │ Metrica veicoli │   R3   │   R1   │    Δ     │ Soglia │  Esito  │
  ├─────────────────┼────────┼────────┼──────────┼────────┼─────────┤
  │ Success rate    │ 22.70% │ 27.38% │ +4.69 pp │ ≥ +2.0 │ ✅ PASS │
  ├─────────────────┼────────┼────────┼──────────┼────────┼─────────┤
  │ Stuck+timeout   │ 46.63% │ 43.18% │ −3.45 pp │ ≤ −2.0 │ ✅ PASS │
  ├─────────────────┼────────┼────────┼──────────┼────────┼─────────┤
  │ Collision       │ 25.56% │ 24.10% │ −1.46 pp │ ≤ +1.0 │ ✅ PASS │
  ├─────────────────┼────────┼────────┼──────────┼────────┼─────────┤
  │ Offroad         │  5.11% │  5.33% │ +0.22 pp │ ≤ +1.0 │ ✅ PASS │
  └─────────────────┴────────┴────────┴──────────┴────────┴─────────┘

  Meccanismo — on-mechanism, sopra il rumore:
  - +45 route-complete in valore assoluto (222→267) su ~stesso numero di episodi → non è effetto-denominatore (a differenza di H3, dove i completamenti erano identici).
  - Il contributo maggiore viene dalla coorte timeout: −27 episodi (−2.71 pp) — esattamente il bersaglio di R1 (incentivo di velocità ora attivo oltre il 30% di rotta). Speed veicolo 11.53→13.04 km/h; la coorte timeout passa a 20 km/h (era 15.6).
  - Canary collisione: non peggiora, migliora −1.46 pp. La preoccupazione "più velocità → più collisioni" non si è verificata — il gate safe_to_push (hazard<0.75) tiene.
  - Chunk SR veicoli (6 chunk, ordine episodi): R3 3.7 22.1 16.6 31.9 34.4 27.6 vs R1 3.1 18.5 23.5 43.2 37.7 38.2. R1 è sopra nei chunk 3-6 e non ha il decadimento tardivo (chunk 6: 38.2 vs 27.6). I chunk 1-2 (R1 sotto) mostrano che un rumore di qualche pp c'è, ma la separazione cumulativa e tardiva lo supera.

  Pedoni (riportati separati). SR 89.06%→85.03% (−4.03 pp), stuck 8.18%→12.41% (+4.23 pp). R1 non tocca _pedestrian_reward. Lettura: probabile rumore run-to-run — la SR pedoni ha oscillato 84.7 / 87.2 / 89.1 / 85.0 nelle ultime 4 run (banda ~4 pp), R3 era il picco alto e R1 è dentro la banda; possibile piccola componente di coupling MARL (veicoli più attivi → ambiente condiviso diverso). Non disambiguabile da una singola run, e non è nel gate (veicolo-centrico) — ma va sorvegliato. Nota: la SR congiunta 55.88→56.21 è piatta e maschera tutto → non usarla (measurement_rules).

  Caveat: A/B single-run, rotte non-paired (bug hash() route seed); final_evaluation_completed=false → nessun dato Town05, come per tutta la serie

### VALUTAZIONE R2 

● Valutazione R2 — run carla_mappo_20260517_164707

    Verifica (ricalcolo da episodes.jsonl):

  ┌───────────────────────────┬─────────────┬──────────────────┬─────────────────┐
  │                           │ R1 (325 ep) │ R2 full (342 ep) │ R2 primi 325 ep │
  ├───────────────────────────┼─────────────┼──────────────────┼─────────────────┤
  │ stuck+timeout (conteggio) │         421 │              415 │             401 │
  ├───────────────────────────┼─────────────┼──────────────────┼─────────────────┤
  │ stuck+timeout (rate)      │      43.18% │           40.45% │          41.13% │
  ├───────────────────────────┼─────────────┼──────────────────┼─────────────────┤
  │ no_wp_steps (media)       │      232.94 │           204.84 │          210.08 │
  └───────────────────────────┴─────────────┴──────────────────┴─────────────────┘

  A episodi pari (975 ep-veicolo entrambi) R2 ha 401 stuck+timeout vs 421 di R1 — 20 in meno, e no_wp_steps −22.86. Il mio confronto del turno scorso (415 vs 421 → "piatto") metteva a paragone due run di lunghezza diversa (1026 vs 975 ep-veicolo): quello era l'errore.
  Lo "stuck+timeout PASS" non è un artefatto di denominatore e l'immobilità non è piatta — R2 ha davvero ridotto l'immobilità di ~2 pp, come da progetto.

  Gate a episodi pari (R2 primi-325 vs R1): SR −2.15 (FAIL), stuck+timeout −2.05 (PASS, reale), collision +0.92 (PASS, sotto canary), offroad +3.28 (FAIL).

  Quindi non tocco i doc: la voce R2 di Codex (CLAUDE.md, AGENTS.md, EXPERIMENT_REGISTRY.md) è già accurata — ha la troncatura a episodi pari e il verdetto corretto. Inserire "artefatto di denominatore" ci metterebbe dentro il mio errore. Non serve nessuna correzione.

  Verdetto R2 corretto

  R2 fallisce il gate, ma per le ragioni giuste: offroad +3.3 pp (regressione reale e strutturale — resta al 6-10% nei chunk dove R1 lo
  azzera) e SR −2.15 pp. La collision è borderline/rumorosa (+1.43 full, +0.92 a episodi pari — straddle del canary). Lo stuck+timeout
  migliora davvero ~2 pp: R2 ha fatto ciò per cui era pensato (meno immobilità), ma la mobilità extra è finita in offroad invece che in
  completamenti. "Retained provisional / not promoted" resta la classificazione corretta e i doc la riflettono bene. La mia
  raccomandazione di revert del turno scorso era basata su un quadro troppo nero — il quadro reale è "effetto misto"; la tua scelta di
  tenerlo provvisorio è difendibile.

  Come procedere

  R2 è un candidato fallito/provvisorio — non va trattato come base validata. Due punti reali ancora aperti, ed entrambi i doc di Codex lo dicono: il rumore run-to-run (ormai dimostrato) e le rotte non-paired (hash seed).

  Mia raccomandazione: confermare R1 con 2 seed prima di proseguire. R1 è il trunk e l'unico PASS pulito; due valutazioni di fila (H3, R2) hanno richiesto controlli equal-episode per non farsi ingannare — è il momento di consolidare R1 invece di impilare. Poi Punto 5 (bugfix route-len, che è un fix d'ambiente, gate a sé).

### VALUTAZIONE P5 (BUG FIX ROUTE PLANNER)

● Ho valutato la run. Il bugfix è confermato in codice (git show 24e072e: l'unica modifica di codice è la one-liner in
  plan_vehicle_route; gli altri 5 file toccati sono solo doc/registry/docx) e i due run_config.json sono identici byte-per-byte → l'A/B isola esattamente il Punto 5.

  1. Integrità

  - 212109: 3174 record = 529 ep × 6, 0 duplicati, 0 episodi malformati, 0 NaN/inf. Integrità perfetta.
  - results.json e episodes.jsonl concordano (1192/1587 route_complete in entrambi) — nessuna discrepanza di tooling.
  - Final evaluation: pending (results.json.evaluation vuoto). Evidenza training-only.

  2. Veicoli — metriche cumulative (ricalcolate da episodes.jsonl)

  ┌─────────────────────┬───────────────────────────┬────────────────────────────────┬───────────┐
  │       Metrica       │ 164707 (R2, bug presente) │ 212109 (Punto 5, bug corretto) │     Δ     │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ Episodi totali      │ 342                       │ 529                            │ +187      │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ SR (route_complete) │ 25.54%                    │ 75.11%                         │ +49.57 pp │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ stuck               │ 23.59%                    │ 17.14%                         │ −6.45 pp  │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ timeout             │ 16.86%                    │ 2.14%                          │ −14.72 pp │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ stuck+timeout       │ 40.45%                    │ 19.28%                         │ −21.17 pp │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ collision           │ 25.54%                    │ 4.28%                          │ −21.25 pp │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ offroad             │ 8.48%                     │ 1.32%                          │ −7.16 pp  │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ route% medio        │ 49.59%                    │ 87.08%                         │ +37.49 pp │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ speed km/h          │ 15.25                     │ 12.28                          │ −2.97     │
  ├─────────────────────┼───────────────────────────┼────────────────────────────────┼───────────┤
  │ step medi/ep        │ 900.4                     │ 573.8                          │ −326.6    │
  └─────────────────────┴───────────────────────────┴────────────────────────────────┴───────────┘

  Conteggi assoluti veicolo — route_complete 262→1192, timeout 173→34, collision 262→68: i cali sono reali (eventi assoluti), non effetto-denominatore.

  SR veicolo per chunk: 164707 → 3.5 / 17.5 / 27.5 / 40.4 / 31.0 / 33.3 (picco ~40 poi decade). 212109 → 10.9 / 70.0 / 91.0 / 93.3 / 94.0 / 92.5 (salita pulita, plateau ~93).

  3. Pedoni (riportati separatamente)

  ┌───────────────┬────────┬────────┬──────────┐
  │    Metrica    │ 164707 │ 212109 │    Δ     │
  ├───────────────┼────────┼────────┼──────────┤
  │ SR            │ 84.41% │ 89.60% │ +5.20 pp │
  ├───────────────┼────────┼────────┼──────────┤
  │ stuck+timeout │ 15.59% │ 10.40% │ −5.20 pp │
  ├───────────────┼────────┼────────┼──────────┤
  │ collision     │ 0.00%  │ 0.00%  │ 0        │
  └───────────────┴────────┴────────┴──────────┘

  4. Gate veicolo: formalmente 4/4 PASS — ma il gate qui non va letto come al solito

  SR +49.57 ≥ +2.0 ✓ · stuck+timeout −21.17 ≤ −2.0 ✓ · collision −21.25 ≤ +1.0 ✓ · offroad −7.16 ≤ +1.0 ✓ · integrità ✓.

  Questo NON è un miglioramento di policy. Network, reward, osservazioni e ottimizzatore sono identici tra le due run: l'unica
  differenza è l'env che genera le rotte. Il +49.57 pp è il cambio di distribuzione del task, non apprendimento.

  Meccanismo: pre-fix plan_vehicle_route accettava qualsiasi rotta ≥ 0.5× target senza limite superiore. Le destinazioni sono scelte a 0.6–1.4× euclidei, ma il percorso A* su strada può essere molto più lungo dell'euclideo → rotte da 40–100m a fronte di un target di 5m. A ~12 km/h una rotta del genere non si chiude nei 1000 step dell'episodio → timeout/stuck per costruzione. Il fix (route_len > target·2.0 → None) le elimina. Le prove che è effetto-distribuzione:
  - episodi 342→529 (+55%) a parità di budget timestep ⇒ episodi più corti;
  - step medi 900→574: pre-fix gli episodi morivano vicino al cap 1000 (rotta non finibile), post-fix chiudono a ~574;
  - route% medio 50→87. Più esposizione su rotte lunghe ⇒ anche più collision/offroad assoluti.

  Le curve per-chunk (salita pulita a 93% vs plateau instabile a ~35%) suggeriscono che il bug degradava anche la qualità del training, non solo la metrica — ma questo singolo A/B non separa "task più facile" da "training più sano". Non rivendicarlo come dimostrato.

  5. Conseguenza importante

  Tutte le run della serie H/R (D2, H1–H3, R3, R1, R2) sono state misurate su questa distribuzione di rotte contaminata. I verdetti single-knob restano direzionalmente plausibili (entrambi i bracci condividevano il bug), ma: i numeri assoluti nel registry sono su distribuzione sbagliata, e R1 in particolare interagisce con la lunghezza rotta (R1 rimuoveva il gate route_completion<0.3: su rotte buggy quel gate restava attivo a lungo, su rotte corrette si disattiva subito) → il +4.69 pp di R1 potrebbe non trasferirsi.

  I pedoni: il bugfix non tocca plan_pedestrian_route_by_distance. Il +5.20 pp è quasi certamente meccanico — episodi congiunti più corti (574 vs 900 step) ⇒ meno tempo per restare stuck/timeout — non un cambio di policy pedonale.

  6. Raccomandazione

  1. Tieni il bugfix (non revertarlo): è una correzione di correttezza che allinea il codice al contratto del docstring [0.5x, 2.0x]. Non è un "candidate promosso" come R1 — nel registry va come env correctness fix, con il +49.57 pp annotato come effetto task-distribution.
  2. 212109 diventa la nuova baseline post-bugfix; ogni A/B futuro gira sulla distribuzione corretta.
  3. R2 è il nodo da sciogliere: è "provvisorio, non promosso" ma è dentro al codice che 212109 ha eseguito, e il suo verdetto era su rotte buggy. L'unico modo pulito è un A/B isolante su distribuzione corretta.
  4. Posso aggiornare CLAUDE.md / AGENTS.md / EXPERIMENT_REGISTRY.md / docx con l'esito e il caveat di contaminazione — confermami tu, non li tocco senza ok.


### VALUTAZIONE O1+O2

• Ho valutato carla_mappo_20260518_152016 da episodes.jsonl. Nessuna modifica ai file.

  Integrità
  | Check | Esito |
  |---|---:|
  | Episodi | 518 |
  | Record | 3108 = 518 × 6 |
  | Duplicati | 0 |
  | NaN/inf | 0 |
  | final_evaluation_completed | false |

  Questa è training-only, non Town05/final eval.

  Config
  La run non è direttamente comparabile come checkpoint/config con 20260517_212109: global_obs_dim passa da 216 a 225, coerente con vehicle obs 47D (O1+O2) invece di 44D. Quindi è una run observation-change from scratch.

  Metriche Veicolo
  | Run | Veh SR | Collision | Stuck+Timeout | Offroad | Route | Speed | no_wp |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 20260518_152016 | 72.46% | 4.83% | 16.92% | 5.79% | 0.840 | 15.74 | 139.15 |
  | 20260517_212109 | 75.11% | 4.28% | 19.28% | 1.32% | 0.871 | 12.28 | 131.66 |

  Vs 212109: SR -2.65 pp, collision +0.54 pp, stuck+timeout -2.36 pp, offroad +4.47 pp.

  Equal-window primi 454 episodi: SR -1.69 pp, collision -0.29 pp, stuck+timeout -2.28 pp, offroad +4.26 pp.

  Quindi O1+O2 non passa il gate: migliora il blocco stuck/timeout, ma perde SR e peggiora troppo offroad.

  Route Diagnostics
  Qui il dato più importante della run:

  | Fonte route veicolo | Quota | SR | Lunghezza media | Sotto 15m |
  |---|---:|---:|---:|---:|
  | legacy_fallback | 76.58% | 73.53% | 27.85m | 0.00% |
  | grp / A* | 23.42% | 68.96% | 12.38m | 75.55% |

  Distribuzione route veicolo reale:

  | Lunghezza route | Quota |
  |---|---:|
  | 7.5-12m | 13.58% |
  | 12-15m | 4.12% |
  | 15-22.5m | 5.60% |
  | 22.5-30m | 73.62% |
  | >30m | 3.09% |

  Tra i soli successi veicolo, 183/1126 = 16.25% sono sotto i 15m dichiarati.

  Questo conferma due cose: A* accetta route sotto target, ma il problema ancora più grosso è che il planner A* fallisce spesso e cade su legacy fallback. Quindi easy=15m oggi in pratica non è “15m”: è una miscela di route A* corte e fallback legacy ~28m.

  Decisione
  
  Non promuoverei 20260518_152016 come nuova baseline. È una buona run in assoluto rispetto alle vecchie R1/R2, ma rispetto alla
  baseline forte 212109 è peggiore su SR e offroad.

  Non procederei nemmeno subito a final eval/O1+O2. Prima sistemerei la parte route:

  1. Fix deterministico hash(ad.agent_id) con seed stabile. (DONE)
  2. Fix planner: provare più destination candidate prima di fallback, non una sola random.
  3. Solo dopo valutare se stringere il lower bound a [1.0x, 2.0x].
  4. Aggiungere anche route_under_target_rate TensorBoard (route_length_ratio < 1.0), perché l’attuale route_too_short_rate misura <0.8x, cioè sotto 12m.

  Sintesi netta: policy forte, critic sano (vf_explained_var=0.966), ma O1+O2 non batte 212109 e la diagnosi route mostra un confound serio. Prossimo passo: route-seed + route-planner fix, poi nuova baseline pulita.


 --- ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  --- 
 
 ## Verifica

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

 1 (H1+H1.1) → 2 (H2) → 3 (R1) → 4 (R2) → 5 (route-len) → 6+7 (O1+O2 insieme); 8 (H3) opzionale dopo H1/H2.
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


#########################################################RECAP##################################################################

### RECAP PLAN 18-05-2026

  Recap Piano EVO Aggiornato

  | Blocco | EVO | Tipo | TensorBoard | Scopo |
  |---|---|---:|---:|---|
  | P2.1 | route_under_target_flag/rate | misura | sì | capire quante route sono < 1.0x target |
  | P2.2 | route_candidate_attempts_configured | misura | sì | sapere se stiamo usando 16/32/64 tentativi |
  | P2.3 | route_candidate_attempts_used | misura | sì | misurare quanti candidati vengono davvero valutati |
  | P2.4 | route_candidate_valid_count | misura | sì | capire se A* trova molte route valide o quasi nessuna |
  | P2.5 | route_candidate_rejected_short_count | misura | sì | quantificare route scartate perché troppo corte |
  | P2.6 | route_candidate_rejected_long_count | misura | sì | quantificare route scartate perché troppo lunghe |
  | P2.7 | route_candidate_no_route_count | misura | sì | capire quando GRP/A* non produce una route |
  | P2.8 | route_planning_latency_ms | misura | sì | misurare costo reale del planner su reset/train |
  | P2.9 | route_target_error_m | misura | sì | distanza assoluta tra target e route ottenuta |
  | P2.10 | route_fallback_flag/rate | misura già prevista | sì | distinguere GRP vs fallback distance-based |

  Decisione operativa

  La strategia resta:

  1. Ripristinare distanze curriculum:
      - easy = 30m
      - medium = 60m
      - hard = 100m
      - Town05 eval = 80m
  2. Correggere fallback veicolo:
      - non più fallback fisso ~30m;
      - fallback coerente con route_distance_m del livello attivo.
  3. Tenere A*/GRP multi-candidate:
      - candidate_attempts = 32 come default iniziale;
      - logging esplicito per capire se è troppo costoso o insufficiente.
  4. Non attivare subito lower bound [1.0x, 2.0x] come vincolo rigido:
      - prima logghiamo route_under_target_rate;
      - se resta alto, allora si valuta il gate P3 con lower bound più severo.

  Gate pratico post-EVO

  Promuoverei la modifica route solo se:

  | Metrica | Gate |
  |---|---|
  | route_fallback_rate | scende o resta spiegabile |
  | route_under_target_rate | scende chiaramente vs 17.7% osservato in 152016 |
  | route_planning_latency_ms | non introduce overhead train ingestibile |
  | Vehicle SR | non peggiora oltre rumore |
  | Collision/offroad | non peggiorano > +1pp |
  | Integrità episodi | sempre 6 record/episodio, 0 NaN/inf |

  Quindi questi log vanno nel piano come prerequisito empirico: senza di loro non possiamo sapere se candidate_attempts=32 sta aiutando, costando troppo, o mascherando ancora route troppo corte/fallback eccessivi.
  
  -> Applicate le EVO route/log in modo chirurgico.

  Modificati:

  - carla_core/envs/route_planner.py: multi-candidate A* con vehicle_route_candidate_attempts, contatori candidati e latency.
  - carla_core/envs/carla_multi_agent_env.py: nuove metriche per-agent in info, incluso route_under_target_flag e route_target_error_m.
  - carla_core/agents/centralized_critic.py: tutte le metriche propagate in TensorBoard custom_metrics.
  - carla_core/training/mappo_runtime.py: tutte le metriche aggiunte a episodes.jsonl.
  - carla_core/configs/multi_agent.yaml: default vehicle_route_min_ratio: 0.5, vehicle_route_max_ratio: 2.0,
    vehicle_route_candidate_attempts: 32.

  Log ora disponibili:
  route_under_target_flag/rate, route_candidate_attempts_configured, route_candidate_attempts_used, route_candidate_valid_count,
  route_candidate_rejected_short_count, route_candidate_rejected_long_count, route_candidate_no_route_count, route_planning_latency_ms,
  route_target_error_m.

  Verifiche eseguite:

  - route_planner.py compile: 3/3 pass
  - carla_multi_agent_env.py + YAML parse: 3/3 pass
  - centralized_critic.py + mappo_runtime.py compile: 3/3 pass
  - CodeReviewGraph get_impact_radius: ok, rischio medium; detect_changes invece è andato in timeout.
  
### COMPORTAMENTO ATTESO

 -> Dal codice appena verificato, il comportamento atteso è questo.

  Target Route

  - levels_path: easy 30m, medium 60m, hard 100m.
  - levels_mixed: easy 30m, medium 60m, hard 100m.
  - levels_traffic: route fissa 60m per tutti i livelli.
  - test/Town05: 80m.

  Vehicle Route Planner
  Per ogni veicolo, se route_distance_m è valorizzato:

  1. Usa seed deterministico:
     traffic.seed + reset_count + crc32(agent_id) via SeedSequence.
  2. Cerca destinazioni candidate:
      - prima spawn point a distanza euclidea [0.6x, 1.4x] del target;
      - se non ne trova, allarga a [0.3x, 2.0x].
  3. Mescola i candidati con RNG deterministico.
  4. Prova fino a vehicle_route_candidate_attempts = 32.
  5. Per ogni candidato:
      - chiama GlobalRoutePlanner.trace_route;
      - calcola lunghezza reale WP-to-WP;
      - accetta solo route dentro [0.5x, 2.0x] del target.
  6. Se trova più route valide:
      - sceglie quella con errore minimo rispetto al target, cioè più vicina a 30/60/100/80m.
  7. Se nessuna route valida viene trovata:
      - va in fallback.

  Caveat Importante
  Il fallback veicolo, nel codice attuale, è ancora la chain legacy basata su:

  route_length_vehicle: 15

  quindi circa 15 * 2m = ~30m.

  Questo significa:

  - su easy 30m: fallback coerente;
  - su medium 60m: fallback accorcia molto;
  - su hard 100m: fallback accorcia moltissimo.

  Quindi il comportamento corretto dipende da quanto scende route_fallback_rate. Se resta alto, la difficoltà route medium/hard viene ancora compressa dal fallback. Ora però lo vedremo chiaramente nei log.

  Log Attesi
  In episodes.jsonl e TensorBoard dovremmo vedere:

  - route_fallback_rate
  - route_under_target_rate
  - route_too_short_rate
  - route_candidate_attempts_configured
  - route_candidate_attempts_used
  - route_candidate_valid_count
  - route_candidate_rejected_short_count
  - route_candidate_rejected_long_count
  - route_candidate_no_route_count
  - route_planning_latency_ms
  - route_target_error_m
  - route_length_ratio

  Interpretazione Attesa
  La run è buona lato route planner se osserviamo:

  - route_fallback_rate basso;
  - route_under_target_rate più basso del 17.7% visto in 152016;
  - route_target_error_m contenuto;
  - route_planning_latency_ms non eccessivo;
  - route_candidate_valid_count > 0 spesso;
  - pochi route_candidate_no_route_count.

  Se invece route_fallback_rate resta alto, allora prima di usare la run come evidenza forte sul curriculum conviene implementare anche il fallback distance-based coerente con route_distance_m.


#########################################################PLAN###################################################################

### NEW PLAN 19-05-2026

### Piano chirurgico gate-driven — veicolo + curriculum + pedoni

     Context

     Run 20260519_001217 (2.94M/3M, integra). Problemi misurati da episodes.jsonl: forgetting (SR medium veicolo 52.6%→38.7%, pedone
     70.7%→40.6% in fase hard); hard force-unlocked a 0.552 del budget col veicolo a 46% SR; reward non normalizzata alla lunghezza
     rotta (collision 6/18/25% su easy/medium/hard). Obiettivo: SR massimale veicoli+pedoni, minima degradazione multi-livello; solo
     evo/fix mirate, niente modifiche obs/architettura.

    Candidati (1 per branch; pipeline sequenziale dal trunk, l'ordine risolve gli overlap di file)

    1. Entropy — train_mappo.yaml:27: entropy_coeff_schedule endpoint 250000→2500000.
	- Risposta alla tua domanda sul dinamismo: sì, è hardcoded e fragile (cambi budget → devi ricordare di aggiornare l'endpoint). Due
		opzioni:
		- (a) Scope minimo — lascia hardcoded ora, documenta la convenzione "endpoint = 0.83 × total_timesteps". Cambio futuro è un
			find/replace.
		- (b) Schema-clean — aggiungi entropy_coeff_schedule_fraction: [[0.0, 0.03], [0.83, 0.005]] in YAML + preprocessor in
			train_carla_mappo.py che lo espande in entropy_coeff_schedule assoluto dopo aver letto total_timesteps. Richiede ~10 righe e una modifica al config schema.
		- Raccomandazione (IMPLEMENTED): (b). È più difendibile in tesi ("schedule scala col budget") e costo basso. Trasforma Entropy in candidato +
			piccolo infra-fix. Se preferisci scope chirurgico, (a).
	2. Ped-route — route_planner.py plan_pedestrian_route_by_distance + _setup_pedestrian_route:1117 + multi_agent.yaml: rifiutare rotta
	   <target×0.5 (specchia il pattern veicolo min_route_ratio), param pedestrian_route_min_ratio: 0.5. (Già implementato nello stack.)
	3. Ped-speed — _pedestrian_reward:2000: banda comfort 0.8–1.8→1.2–2.6 m/s.

	4. P1 — curriculum_batch_manager.py:565 + get_episode_level:799: rimuovere l'esclusione di easy; floor min_probabilities easy≥0.10 /
		medium≥0.20 in fase hard (rehearsal).
	5. P2 — curriculum_batch_manager.py _balanced_policy_success_details:422: mean(SR)→min(SR); unlock gated sulla policy più debole.
	6. P3 — curriculum_batch.yaml unlock_criteria.hard: force_unlock_global_share_cap 0.55→0.70.
	- Risposta alla tua domanda sul budget residuo: 
		obiezione legittima. A 0.70 restano solo 30% del budget totale per hard. Con hard_min_share: 0.35 configurato come dynamic floor post-unlock (non share cumulativo), hard riceve 30% × 0.35 = ~10.5% del budget totale di pratica hard. Possibilmente troppo poco se la policy è ancora debole. 
		Trade-off:
        - 0.55 (attuale): unlock prematuro, vehicle a 46% SR sul corner → forgetting di easy/medium (osservato)
		- 0.70 (proposto): più consolidamento, ma hard sotto-praticato se la policy non raggiunge competence
		- 0.60–0.65 (compromesso): scelta più conservativa, mai testata
		- Logica del piano: il cap è una rete di sicurezza, non il path nominale. Con P1 (rehearsal floors) + P2 (min-SR gate) attivi,
		  l'unlock per competence dovrebbe arrivare PRIMA del cap, rendendo 0.70 raramente attivato. Solo se il fallback scatta il valore conta. Quindi 0.70 è ragionevole se P1+P2 fanno il loro lavoro; rischioso se la competence non si sviluppa mai.
		- Suggerimento: mantieni 0.70 per il primo run di Block 4 e instrumenta il log per registrare se il force-unlock è effettivamente
		  scattato. Se è scattato, valutare 0.60. Se non è scattato (competence-unlock arriva prima), 0.70 era irrilevante.
	7. R-norm v2 — carla_multi_agent_env.py _vehicle_reward: abbandonato design v1 (divisione per route_wp_count ha causato collasso
	   magnitudine). Design corretto: length_factor = TARGET_ROUTE_WP / route_wp_count (TARGET_ROUTE_WP=30, midpoint
	   easy(15)/medium(35)/hard(60)) applicato a TUTTI i per-step shaping (positivi e negativi); terminale -500 invariato. Condizionale a
	   Block 4: skip se P1+P2+P3+Ped-route appiattiscono il gradiente collision easy→hard; procedi se persiste.
	
     Overlap file: P1+P2 (curriculum_batch_manager.py); R-norm/Ped-route/Ped-speed (carla_multi_agent_env.py) → ordinati, mai paralleli.
      
	-> Ordine: Entropy→(Ped-route→Ped-speed)→(P1→P2→P3)→R-norm.

     Gate (per candidato, vs baseline 20260519_001217; fallito → revert solo quello)

     - R-norm/Entropy/Ped-speed: <gate_policy> per-livello + vf_explained_var≥0.8 per fase.
     - P1/P2/P3: final-eval per-scenario (il training non misura il forgetting di easy): hard SR +≥2pp, easy/medium SR ≥−2pp
     (no-forgetting), Town05 non peggiora.
     - Ped-route: diagnostico — pedoni route_length_ratio<0.5 → ~0%.
     - Sempre: 6 record/ep, 0 NaN/inf.

     Verifica

     Per branch: python -m compileall sui file toccati + git diff --check + integration-check ≤3 iterazioni. Le run le lancia l'utente;
     metriche ricalcolate da episodes.jsonl (+results.json per la final-eval).
	 
	 
### AFTER GATE EVO PLAN

  2. Non usare la run carla_mappo_20260519_001217 come verdetto finale
      - È meccanicamente healthy.
      - Però è contaminata lato pedoni dal bug sulle rotte sidewalk troppo corte.
      - Resta utile per diagnostica vehicle/curriculum, ma non thesis-grade per pedoni o confronto curriculum vs mixed.
  3. Rifare una baseline pulita MLP 47D
      - Stessa architettura attuale: no Attention, no PopArt, no GNN.
      - Serve come riferimento corretto dopo il fix pedoni.
      - Senza questa baseline, Attention/PopArt/GNN non sono interpretabili.
  4. Run esplorative da 300K solo come screening
     Ordine consigliato:
      - MLP 47D clean baseline
      - PopArt only
      - Attention critic only
      - Attention + PopArt
      - GNN/GAT only
      - GNN/GAT + PopArt
      - eventuale GNN/GAT + Attention/PopArt solo se i risultati precedenti giustificano il costo
  5. Non scegliere “best metrics” da una singola 300K
      - Una run 300K serve per eliminare candidati deboli.
      - Per promuovere una variante servono almeno 3 validazioni/seed.
      - Solo dopo si lancia una long run comparativa.
  6. Priorità decisionale
      - Primario: vehicle hard SR.
      - Secondario: ridurre stuck+timeout.
      - Canary: collision/offroad non devono peggiorare.
      - Easy/medium non devono degradare troppo, altrimenti la policy diventa specializzata su hard.
  7. Gate proposti
      - Integrità: 6 record per episodio, 0 malformed, 0 incompleti.
      - Route quality: veicoli route_ratio < 0.5 = 0; pedoni idem dopo fix.
      - Vehicle hard: SR almeno +3 pp vs baseline.
      - Stuck+timeout hard: almeno -2 pp.
      - Collision/offroad: non oltre +1 pp.
      - Easy/medium SR: degradazione massima circa -2 pp.
  8. Valutazione architetture
      - PopArt: priorità alta, perché normalizza target critic multi-scala; utile ma va verificato perché vf_clip_param=1e6 è già
        attivo.
      - Attention: priorità alta-media, plausibile per CTDE multi-agent.
      - GNN/GAT: priorità più bassa, blast radius alto e solo 6 agenti fissi; da testare dopo Attention/PopArt, non prima.

  Conclusione operativa: screening 300K delle architetture, poi 3-seed validation, poi long run comparativa


#########################################################BUGS###################################################################

### NEW BUG FOUND [19-05-2026] (FIX INTO NEW PLAN 19-05-2026)

• Sì. Confermo: è un bug reale del routing pedonale, non solo un caso isolato.

  Evidenza Da Run Live
  Snapshot 2026-05-19 18:13:02, run carla_mappo_20260519_001217.

  | Scope | Pedoni con route_length_ratio < 0.5 | route_too_short_flag | SR pedoni | Stuck+Timeout |
  |---|---:|---:|---:|---:|
  | Totale pedoni | 3183/8784 = 36.24% | 49.61% | 67.83% | 31.86% |
  | Easy | 635/3063 = 20.73% | 32.45% | 88.80% | 11.17% |
  | Medium | 1219/3243 = 37.59% | 52.91% | 69.13% | 30.68% |
  | Hard | 1329/2478 = 53.63% | 66.51% | 40.19% | 59.00% |

  Confronto: veicoli con route_length_ratio < 0.5 = 0/8784. Quindi il lower-bound è rispettato per i veicoli, non per i pedoni.

  Causa Nel Codice

  - /C:/Users/kyros/OneDrive/Desktop/TESI_PROJECT_traffic-rl-curriculum-vs-mixedbatch/traffic-rl-curriculum-vs-mixedbatch/carla_core/
    envs/route_planner.py:253 plan_pedestrian_route_by_distance: concatena waypoint sidewalk finché può, ma se la chain finisce prima
    del target ritorna comunque wps se non vuoto.
  - /C:/Users/kyros/OneDrive/Desktop/TESI_PROJECT_traffic-rl-curriculum-vs-mixedbatch/traffic-rl-curriculum-vs-mixedbatch/carla_core/
    envs/carla_multi_agent_env.py:1117 _setup_pedestrian_route: accetta qualsiasi wps con len(wps) >= 2, senza validare
    route_optimal_length / target.
  - /C:/Users/kyros/OneDrive/Desktop/TESI_PROJECT_traffic-rl-curriculum-vs-mixedbatch/traffic-rl-curriculum-vs-mixedbatch/carla_core/
    envs/route_planner.py:206 invece per veicoli calcola lower = target_distance_m * min_route_ratio e rigetta route_len < lower.

  Impatto Sul Verdetto
  La mia valutazione precedente va corretta così:

  | Aspetto | Verdetto aggiornato |
  |---|---|
  | Health meccanica run | ancora OK: JSONL integro, curriculum avanza, nessun crash |
  | Vehicle hard | interpretabile |
  | Pedestrian hard | confonded / non interpretabile pulitamente |
  | Degradazione pedoni su hard | non si può attribuire solo alla difficoltà; il task pedonale hard è contaminato da route troppo corte
  |

  Quindi: la run può restare utile come diagnostica veicoli e curriculum live, ma non la userei come evidenza finale thesis-grade per i
  pedoni o per il confronto curriculum vs mixed-batch senza correggere il routing pedonale e rilanciare.

  Patch Proposta, Non Applicata
  File impattati:

  - carla_core/envs/route_planner.py
  - carla_core/envs/carla_multi_agent_env.py
  - opzionale ma consigliato: carla_core/configs/multi_agent.yaml

  Modifica mirata:

  - aggiungere pedestrian_route_min_ratio: 0.5;
  - fare rifiutare a plan_pedestrian_route_by_distance rotte con lunghezza < target * min_ratio;
  - passare il parametro da _setup_pedestrian_route;
  - lasciare logging/flag per quantificare quante rotte vengono scartate.

  Rischio sperimentale: alto se applicata mid-run, perché cambia la distribuzione del task. Non va applicata dentro 20260519_001217
  continuandola come se nulla fosse.

  Verifica prevista:

  - compileall;
  - git diff --check;
  - mini-check su JSONL post-run: pedoni route_length_ratio < 0.5 deve scendere a 0 per sidewalk_distance;
  - nuova run breve sanity, poi restart full curriculum 47D pulito.


##############################################################Q&A#########################################################################

### Q&A 20-05-2026

> Vanno testate 1 ad 1 con run esplorative da 300K o posso aggiungerle tutte in blocco e lanciare la long run?                          

● Nessuna delle due opzioni è ottimale per tutti i 7. Si dividono in due insiemi con orizzonti di osservabilità diversi, e la scelta
  dipende da quello:

  Osservabili in short-horizon (300K easy-locked) — il meccanismo agisce per-step dal primo episodio:
  - R-norm — cambia la scala del reward veicolo. Da testare individualmente perché ridefinisce il plateau (sotto la nuova
  normalizzazione il reward totale di rotta scende da ~`N×100` a ~100; il critico deve ribilanciarsi). 300K easy-only è il minimo; 500K
  è più sicuro per vedere il plateau sotto la nuova scala.
  - Ped-speed — cambia il reward pedone per-step. 300K easy-locked vede se i pedoni convergono verso la nuova banda 1.2–2.6 m/s.
  Sufficiente.
  - Entropy — il riscalamento dell'endpoint agisce sull'esplorazione globale ma il decadimento iniziale è osservabile a 300K (entropia
  media, KL, vf_explained_var). 300K sufficiente per confermare il meccanismo; effetto su SR no.

  NON osservabili in short-horizon (richiedono long run) — il meccanismo si attiva solo dopo l'unlock di hard, che nella tua run
  precedente è arrivato a 1.66M timesteps:
  - P1 — il floor di rehearsal si attiva post-unlock di hard + dopo la probation. A 300K non si tocca.
  - P2 — min(SR) cambia la metrica di unlock, ma in 300K non c'è alcun unlock candidato.
  - P3 — la cap a 0.70 farebbe scattare il force-unlock a 2.1M (su 3M). A 300K non si vede nulla.

  Raccomandazione (3 run totali):
  (DONE) 1. EVO/entropy — 300K easy-locked. Gate diagnostico: entropia decresce monotona, KL stabile. (PASS)
  (TODO) 2. EVO/ped-route+ped-speed — 300K easy-locked. Gate: pedone speed media verso 1.5–2.2 m/s, SR ped ≥−2pp.
  3. P1+P2+P3 stackati — 3M --difficulty path. Mergi i 4 branch in un branch EVO/curriculum-stack, una sola long run. Il
  confounding tra P1/P2/P3 è inevitabile (testarli individualmente costerebbe 9M); però condividono lo stesso bersaglio (forgetting +
  premature unlock) e hanno logging diagnostico distinguibile (unlock-event reason, min_probabilities, balanced_sr vs nuovo min_sr
  registrato).
  4. R-norm v2 (Block 3: skip se P1+P2+P3+Ped-route appiattiscono il gradiente collision easy→hard; procedi se persiste.)

  -> Ordine: Entropy→(Ped-route→Ped-speed)→(P1→P2→P3)→R-norm.

  Non merge su tutti e 7 in una sola long run: avresti un solo numero finale e zero capacità di attribuzione tra reward-side
  (R-norm/Ped-speed/Entropy) e curriculum-side (P1/P2/P3). Se la run regredisce, è epistemologicamente peggio di adesso.


### Q&A 20-05-2026

> Non so se lasciare la difficoltà dei path livelli as-is o riportarla a 15-30-60 metri, con la forbice dinamica, in cui può         
  prendere solo valori maggiori o uguali al lower bound o vicini, ma mai superiori all'upper bound, cosa mi consigli?                

● Sul merito: as-is vs 15/30/60 con forbice dinamica

  Tre opzioni reali, non due:

  Opzione: A. As-is 30/60/100
  Pro: nessuna variabile cambia in corsa; pulizia scientifica
  Contro: easy=30m è già duro per veh policy 47D (stuck 84% nei primi 60 ep); rischio di non sbloccare medium e bruciare budget
  ────────────────────────────────────────
  Opzione: B. Punto fisso 15/30/60
  Pro: riduce sparse-reward bottleneck su easy; salti geometrici 2×/2× (più clean del 2×/1.67× attuale)
  Contro: introduce un'altra variabile rispetto al baseline 47D 20260518_195947; non testa generalizzazione intra-stage
  ────────────────────────────────────────
  Opzione: C. Forbice [L,U] con bias al lower bound
  Pro: diversità intra-stage → migliore transfer Town05; bridge naturale tra livelli; allineata a best practice curriculum-RL
  Contro: aggiunge un knob; rende la windowed-SR "soglia di unlock" un target mobile; più difficile attribuire risultati al singolo
    asse

  Cosa ti consiglio

  Non cambiare in corsa il run 20260520_133747. Stoppare/riavviare ora costa ore di compute e rende non comparabile la run;
  lasciala arrivare almeno a ~150k env-step e guarda:
  - vehicle SR windowed su easy (cresce?)
  - curriculum_level corrente (resta su easy o sblocca?)
  - vf_explained_var (il critic regge?)

  Per la PROSSIMA iterazione, ordine di priorità:

  1. Prima: passa a 15/30/60 punto fisso (Opzione B). Singolo cambio rispetto al baseline 47D, salti geometrici puliti, easy=15m è
  dentro il range in cui sai che la policy 47D riesce a chiudere (pilota easy-locked plateau 79%). Mantieni windowed-SR come gate.
  2. Solo dopo che B ha verificato che il meccanismo di unlock funziona end-to-end, introduci la forbice (Opzione C) come
  final-comparison config per il confronto curriculum-vs-mixed-batch. È lì che la diversità intra-stage paga davvero (transfer
  Town05).

  Sulla forbice specifica che proponi ("≥ lower o vicini, mai > upper"): tecnicamente è una distribuzione troncata con moda al lower
  bound (es. esponenziale troncata, o Beta sbilanciata a sinistra). Concretamente:
  - easy: clip(Exp(λ=1/5) + 15, 15, 30) → moda 15m, coda fino a 30m
  - medium: clip(Exp(λ=1/8) + 30, 30, 60) → moda 30m, coda fino a 60m
  - hard: clip(Exp(λ=1/15) + 60, 60, 100) → moda 60m, coda fino a 100m

  Così ogni livello sovrappone il floor con il ceiling del precedente, smussando le transizioni. Ma di nuovo: questo è il passo 2,
  non da introdurre ora.

  Sintesi operativa:
  - Adesso: lascia il run, niente cambi.
  - Prossimo run: 15/30/60 punto fisso, allinea CLAUDE.md ↔ levels.yaml.
  - Run finale di tesi: forbice troncata sul setup vincente.


### Q&A 20-05-2026 (P2)

> Ricordami come avviene la promoption tra livelli adesso, usiamo la media minima tra le due SR veicoli e pedoni?                       

  Searched for 2 patterns, read 1 file (ctrl+o to expand)

● Risposta breve: no, non è il minimo — è la media aritmetica tra vehicle_policy e pedestrian_policy (entrambe in policy_ids).

  Logica attuale di unlock (curriculum_batch_manager.py:391-475):

  1. Metrica usata: balanced_sr = mean(window_success_rate(vehicle), window_success_rate(pedestrian)) — riga 422, semplice sum/len.
  2. Gate hard per-policy DISATTIVATO: require_policy_success: false in curriculum_batch.yaml:11. Quindi il check per-policy
  (_policy_gate_details, righe 358-389) che richiederebbe window_success_rate >= threshold su ciascuna policy non viene applicato. Resta
   solo la media.
  3. Condizioni per unlock "competence_unlocked" (riga 470-475):
    - tracker.level_timesteps >= min_budget_share * total_budget_timesteps (es. medium serve ≥20% budget)
    - tracker.window_full (window di 50 episodi piena sul livello aggregato)
    - balanced_sr >= success_rate_threshold
  4. Fallback "pressure_cap_unlocked" (riga 461-465): se si supera force_unlock_global_share_cap, si sblocca a forza ignorando la SR.

  Implicazione per il tuo caso: con pedoni a ~95% SR e veicoli a ~50%, la media è ~72% — sopra molte threshold tipiche (0.5-0.6), quindi
   il veicolo da solo non bloccherebbe la promozione. Se vuoi che il veicolo (policy debole) sia il vincolo, le opzioni minime invasive
  sono:

  - Mettere require_policy_success: true → riattiva il gate hard per-policy (de facto un min(SR_v, SR_p) >= threshold + collision
  check).
  - Cambiare l'aggregazione in _balanced_policy_success_details da mean a min (one-liner alla riga 422).
  - Pesare la media (es. 0.7*veicolo + 0.3*pedone) — più invasivo.

  Vuoi che proceda con una di queste? (Solo proposta, non eseguo finché non confermi.)


##########################################################################################################################################


### Q&A 18-05-2026

> Qual è il vantaggio di avere A* e la logica che abbiamo fiaxato al posto di avere una route statica e crescente per tutti e 3 i livelli?

● Domanda di design sperimentale — rispondo confrontando le due opzioni rispetto alla tua research question (curriculum vs mixed-batch, generalizzazione Town03→Town05).

  Cosa fa oggi plan_vehicle_route

  Per ogni episodio/reset: sceglie uno spawn point casuale a ~`target_distance_m euclidei, traccia un percorso con **A\* sul 
  road-graph** (GlobalRoutePlanner.trace_route), valida la lunghezza nel contratto [0.5x, 2.0x]del target, e tiene il candidato più vicino al target. I 3 livelli varianoroute_distance_m` (30/60/100m), quindi la difficoltà è una distribuzione di route che cresce, non una route fissa che cresce.

  L'alternativa che proponi — una route statica (geometria fissa) per ciascuno dei 3 livelli — ha un solo vantaggio reale: determinismo banale e zero fallback. Tutto il resto è a sfavore.

  Vantaggi di A* + logica di validazione

  1. Evita la memorizzazione della traiettoria — è un punto ottimale per la tesi. (più generalizzazione)
  Con una route statica la policy può raggiungere SR alta imparando a memoria una sola traiettoria ("curva a sinistra qui, accelera lì") senza imparare a guidare. A quel punto:
  - il test di generalizzazione Town03→Town05 misurerebbe solo overfitting;
  - il confronto curriculum vs mixed-batch diventerebbe "scheduling di memorizzazione di 3 traiettorie" invece di "l'ordinamento di difficoltà aiuta a imparare una policy generalizzabile". La domanda di ricerca verrebbe svuotata.
  A* per-episodio espone l'agente a una distribuzione di incroci, curve, curvature, rettilinei → costringe a una competenza
  trasferibile.

  2. La difficoltà è un asse pulito, non confuso con una geometria.
  Con route statiche, "hard" = quella curva difficile che hai scelto tu. Con A* + target distance, "hard" = route statisticamente più lunghe → più incroci, più punti di decisione, più esposizione al traffico. La difficoltà è monotòna e non confusa con un layout specifico — condizione necessaria perché il curriculum sia un esperimento controllato.

  3. Validità del road-graph gratis.
  A* sulla topologia CARLA garantisce corsie percorribili, sensi di marcia, connettività degli incroci. Una route statica andrebbe verificata a mano su ogni mappa.

  4. Multi-agent: i 3 veicoli RL devono avere route diverse.
  3 veicoli sulla stessa route statica si accodano/collidono banalmente. A* da ogni spawn dà percorsi spazialmente distinti — requisito per un setup multi-agente sensato.

  5. Le metriche di gate hanno senso solo su una distribuzione.
  SR/collision/stuck misurati su route variate sono onesti; su una route memorizzata il gate +2pp SR non significherebbe nulla.

  Il costo di A* — ed è esattamente ciò che abbiamo "fixato"

  A* verso destinazione casuale può produrre detour molto più lunghi della distanza euclidea. I costi e le relative mitigazioni:

  - Route infattibili → il contratto [0.5x, 2.0x] (bugfix Punto 5) scarta le route > 2x target, non finibili nei 1000 step. Senza, SR strutturalmente depressa.
  - Non riproducibilità → il fix del route-seed (SeedSequence al posto di hash()) rende le run A/B route-paired. Una route statica sarebbe deterministica gratis; A* lo diventa solo con seeding corretto.
  - Fallback → quando A* non trova route valide entro 32 candidati ricade su legacy_fallback (osservato al 16.4% nella run
  20260518_195947). Una route statica non fallisce mai — ma questo è il prezzo accettabile della varietà, ed è proprio il punto aperto da diagnosticare prima della long run.

  In sintesi

  La route statica vince solo su semplicità/determinismo. A* + validazione vince su tutto ciò che la tua tesi richiede:
  generalizzazione, difficoltà come asse controllato, multi-agente sensato, metriche oneste. I "fix" che hai messo ([0.5x,2.0x],
  route-seed) non sono toppe a un design fragile — sono il modo corretto di pagare il costo di A* mantenendone i benefici. Con route statiche non avresti niente da fixare, ma non avresti nemmeno un esperimento valido.


### Q&A 18-05-2026

> Quale trigger unlock scegliere tra SR windowed e cumulativa?

● Raccomandazione: SR windowed per il trigger di unlock

  È la scelta più difendibile e più solida per il tuo obiettivo (metrica massimale a fine budget). Ecco il ragionamento.

  Perché è più difendibile

  L'unlock è un segnale di controllo, non una metrica riportata. Un controllore deve riflettere la competenza attuale. La SR cumulativa è un integratore in ritardo: per costruzione media per sempre i fallimenti del cold-start, quindi sotto-stima in modo strutturale e permanente l'abilità corrente. Il pilota lo dimostra: 86% windowed (competenza reale) vs 67% cumulativo. Usare un segnale con bias-basso noto per decidere "l'agente è pronto?" risponde alla domanda sbagliata — "è stato bravo in media dalla nascita" invece di "è bravo ora".

  Non c'è conflitto con la measurement_rule di CLAUDE.md ("primary aggregation: cumulative"): quella regola vale per il reporting della run. Il reporting resta cumulativo; il trigger di controllo usa windowed. Sono due funzioni diverse e vanno tenute distinte.

  Perché è più solida per "metrica massimale a fine budget"

  1. Efficienza di budget. Il gradiente su un livello padroneggiato è ≈ 0. La SR cumulativa tiene l'agente su easy molto dopo che il gradiente si è esaurito (pilota: easy è risolto a ~250k, ma la cumulativa direbbe ancora "continua su easy"). Ogni step su easy oltre la mastery è budget sprecato → meno budget per medium/hard → competenza finale più bassa. La windowed sposta il focus quando la competenza è reale.
  2. La cumulativa ha un failure mode su long run. Con abbastanza fallimenti iniziali, la cumulativa potrebbe non superare mai la soglia entro il budget: l'unlock cadrebbe sempre sul force_unlock_global_share_cap. Risultato: la success_rate_threshold (la riga che hai selezionato) diventa codice morto e il curriculum degenera in uno schedule a timestep fissi. La windowed fa effettivamente scattare l'unlock competence-based — cioè dà senso al metodo curriculare.

  L'obiezione "windowed è rumorosa" — è già mitigata

  La windowed (finestra 50 episodi) potrebbe scattare su una finestra fortunata. Ma è una preoccupazione triplamente coperta:
  - finestra da 50 episodi → la maggior parte del rumore è già smorzata;
  - l'unlock richiede anche level_timesteps >= min_budget_share (:468) → una finestra fortunata troppo precoce non sblocca prima della spesa minima;
  - probation_blocks_after_unlock = 2 → un unlock leggermente anticipato è già cuscinato (riduce la pressione di campionamento sul nuovo livello per 2 blocchi).

  La SR cumulativa, invece, ha un downside (il lag strutturale) senza alcuna mitigazione.

  Nota tecnica

  Il codice ha già la windowed: tracker.window_success_rate è usata in _policy_gate_details (:377,384), ma quel gate è spento
  (require_policy_success: false). Il path competence_unlocked attivo (:468-472) legge invece cumulative_success_rate via
  _balanced_policy_success_details (:405). Il cambio è narrow: far leggere a quel path la window_success_rate bilanciata invece della cumulativa. force_unlock_global_share_cap resta come backstop.

  In sintesi: adotta windowed. Se vuoi un margine extra di robustezza si può aggiungere un guard "soglia sostenuta per 2 blocchi
  consecutivi", ma con min-budget-gate + probation già presenti la windowed semplice è sufficiente e più pulita.


### Q&A

> Dove trovo i log di TensorBoard?

● TensorBoard, Dove sono i log (≠ cartella del progetto)

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
  - Senza TB: progress.csv nella stessa dir contiene tutte le 238 colonne (stessi nomi della tabella sopra) → apribile con pandas se preferisci grafici tuoi.
  - Opzionale (fuori dal piano): il timeseries di results.json non include vf_explained_var — il dato c'è già nel result dict
  (result["info"]["learner"]…, vedi train_carla_mappo.py:284). Aggiungere una riga al timeseries.append (train_carla_mappo.py:1105) lo
  porterebbe nella pipeline del progetto, così compare_results_carla.py può usarlo senza TB. Dimmi se vuoi che lo aggiunga al piano come punto extra.
  

### Q&A
  
  > Domanda: Avrebbe senso aumentare il tempo totale degli episodi (tipo da 1000 a 1500) al fine di recuperare tutti quei veicoli che  vanno in timeout e di conseguenza implementare una logica mirata che vada a troncare l'episodio nel momento in cui il veicolo è stuck (ad esempio se continuous_route_progress < 0.1 e no_wp_steps > 150, o altri valori in combo [ma senza inserire troppo rumore o rigidità]) intercettandolo in tempo così da non aggravare il train? -rispondi criticamente ed in modo empirico in base ai dati che abbiamo a disposizione su episode.jsonl 

● Ho ricalcolato dalla baseline completa 211642 (960 record veicolo, deduplicati). Il dato che ribalta l'impostazione della domanda è questo:

  ┌────────────────┬─────┬────────────────────────┬─────────────────────────┬─────────────────┬─────────────┐
  │ Coorte veicolo │  n  │ route_compl. (mediana) │ no_wp_steps (med / p90) │ speed_kmh (med) │ step_count  │
  ├────────────────┼─────┼────────────────────────┼─────────────────────────┼─────────────────┼─────────────┤
  │ timeout        │ 227 │ 0.53                   │ 5 / 197                 │ 16.2            │ 1000        │
  ├────────────────┼─────┼────────────────────────┼─────────────────────────┼─────────────────┼─────────────┤
  │ stuck          │ 293 │ 0.07                   │ 439 / 1000              │ 0.25            │ 1000        │
  ├────────────────┼─────┼────────────────────────┼─────────────────────────┼─────────────────┼─────────────┤
  │ route_complete │ 193 │ 1.00                   │ 0                       │ 15.7            │ 862 (media) │
  └────────────────┴─────┴────────────────────────┴─────────────────────────┴─────────────────┴─────────────┘

  La coorte timeout NON è una coorte di veicoli "stuck". no_wp_steps mediana = 5 (avanzano un waypoint 0.25 s prima che scada il clock),83 % sotto 100, velocità terminale mediana 16 km/h ≈ identica ai successi (15.7). Il rilevatore di stuck del simulatore già separa i due gruppi: i 293 "stuck" sono i veri fermi, i 227 timeout sono il residuo che si stava muovendo. Questo cambia la risposta a entrambe le domande.

  Q1 — Alzare max_steps 1000 → 1500: criticamente NO (o solo condizionato)

  Tetto meccanico, onesto: solo 47/227 timeout (≈21 %) hanno route_completion ≥ 0.667 e no_wp_steps < 100 → realisticamente recuperabili entro 1500 step. È ~+5 pp SR, e pure ottimistico (assume che mantengano il rate medio per altri 500 step).

  Perché comunque non farlo:

  1. È un cambio di metro di misura, non un miglioramento di policy. Con max_steps=1500 la SR sale a policy identica byte-per-byte.
  CLAUDE.md <do_not_infer>: «non trattare miglioramenti di tooling/misura come miglioramenti di policy». Passerebbe pure il gate
  numerico (+SR, −timeout) → un pass di Potemkin.
  2. Rompe la comparabilità di tutto il track record. Il timeout è definito da max_steps. C0/C1/D2/D3 e la baseline curriculum sono tutti misurati a 1000. Per una tesi che confronta comportamento curriculum vs mixed-batch, cambiare il clock a metà studio invalida i confronti.
  3. 1000 step NON sono il vincolo per una rotta easy vera. 15 m a 15 km/h ≈ 70 step; anche a 5 km/h ≈ 216. Se il successo mediano impiega 1000 step per "15 m", o le rotte non sono 15 m (bug route planner — Punto 5 del piano) o la policy striscia (R1/H2). Estendere il clock cura il sintomo.
  4. Anche i 47 "recuperati" sono successi di bassa qualità: un veicolo che serve 1500 step per 15 m si comporta male. Gonfi la SR mascherando il comportamento lento — proprio ciò che la tesi deve misurare.

  Gate decisionale: rivalutare max_steps solo dopo Punto 5 (cap lunghezza rotta) + R1 + H2, rimisurando la coorte timeout su rotte a lunghezza verificata. Se lì i timeout persistono con no_wp<100 e rc alto → allora un cambio documentato e con baseline ri-basata è giustificato. Adesso il numero 47 è confuso dal bug delle rotte.

  Q2 — Troncamento anticipato stuck (continuous_route_progress<0.1 & no_wp_steps>150): NO, è D3

  Questo è D3, già rigettato empiricamente. D3 (run 20260514_190424, trigger no_wp_steps>=300 & route<0.3 & hazard<0.75): SR −2.90 pp,
  stuck+timeout +7.07 pp, gate fallito, reverted (registro CLAUDE.md). La tua predicato con no_wp_steps>150 scatta prima di >=300 → più aggressivo di D3, atteso peggiore.

  Scan empirico del tuo predicato sulla baseline: matcha 183 veicoli = 135 stuck + 27 offroad + 21 collision, ZERO timeout. Cioè:
  - Non tocca affatto la coorte timeout che vuoi recuperare (i timeout hanno continuous_route_progress mediana 0.53, mai < 0.1). Il tuo Q2, per costruzione, non interseca il tuo Q1.
  - Accorcia solo episodi già persi, e ri-etichetterebbe 21 collisioni + 27 offroad come "stuck" → distorce collision/offroad rate.

  Perché non può aiutare (meccanismo): troncare non recupera nulla — converte un episodio lungo in uno "stuck" corto, rimuove la coda
  dell'episodio dal training, e azzera la possibilità che il rumore di esplorazione campioni un recupero. Sposta la distribuzione dati
  verso stati facili/iniziali → la policy peggiora proprio sugli stati difficili: è esattamente l'impronta del +7.07 pp stuck+timeout di D3. Non è un problema di soglie o "rumore" da tarare — il danno è strutturale (bias di distribuzione), nessun valore di soglia lo
  risolve.

  L'alternativa Markov-corretta è già nel piano: Punto 6 (O1) rende no_wp_steps e loop_flag osservabili → l'agente impara a reagire allo stuck, invece che far amputare l'episodio dall'env.

  In sintesi

  Tieni max_steps=1000 e non aggiungere il candidato Q1/Q2: sono un cambio di misura + un candidato già rigettato, e nel piano combinato si contraddicono (Q1 allunga +50 % di step sprecati anche sui 293 veri stuck, Q2 serve a recuperarne una parte). Il piano già attacca timeout/stuck alla causa: Punto 5 (bug rotte — de-confonde l'interpretazione del timeout), R1 (incentivo velocità oltre il 30 %), H2 (orizzonte), O1 (osservabilità). La run H1 175921 è ancora in corso — rimisura la coorte timeout dopo che H1/Punto 5/R1/H2 sono atterrati, prima di toccare il clock.


  R3 — calibrazione della penalità collisione

  Anatomia _vehicle_reward (verificata dal codice):

  ┌──────────────────────────────────────┬─────────────────┬───────────┐
  │              Componente              │     Valore      │   Riga    │
  ├──────────────────────────────────────┼─────────────────┼───────────┤
  │ Waypoint raggiunto                   │ +100 / waypoint │ 1731      │
  ├──────────────────────────────────────┼─────────────────┼───────────┤
  │ Avvicinamento al wp                  │ ±4 / metro      │ 1743      │
  ├──────────────────────────────────────┼─────────────────┼───────────┤
  │ Collisione                           │ −50 one-shot    │ 1748      │
  ├──────────────────────────────────────┼─────────────────┼───────────┤
  │ Off-lane / idle / sterzo / anti-loop │ O(0.1–1.0)/step │ 1750–1811 │
  └──────────────────────────────────────┴─────────────────┴───────────┘

  Il problema, in numeri. La penalità collisione −50 = metà di UN waypoint (+100). Il calcolo che fa l'agente:

  "Guido aggressivo → prendo 1 waypoint → mi schianto" = +100 − 50 = +50 netto positivo.

  Schiantarsi conviene se prima agguanti ≥1 waypoint. È strutturale — ed è esattamente perché H2 è fallito: ha propagato su orizzonte
  lungo una penalità che perde ogni trade-off locale. Il problema non era γ, era la magnitudo.

  Perché −500 (raccomandato):

  ┌───────────────┬──────────────┬─────────────────────┬────────────────────────────────────────────────────────────────────────
  │   Penalità    │ = N waypoint │  "1 wp poi crash"   │                               Effetto                                  
  ├───────────────┼──────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────
  │ −50 (attuale) │ 0.5 wp       │ +50 → conviene      │ crash più economico di 1 unità di progresso
  ├───────────────┼──────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────
  │ −500 ✅       │ 5 wp         │ −400 → non conviene │ crash = danno reale; servono ≥5 wp per pareggiare
  ├───────────────┼──────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────
  │ −2000         │ 20 wp        │ −1900               │ > valore intera rotta (~15 wp, route_length_vehicle:15) →iper-difensività
  └───────────────┴──────────────┴─────────────────────┴────────────────────────────────────────────────────────────────────────

  −500 capovolge il segno del trade-off locale (+50 → −400) con un fattore 10×: abbastanza per cambiare comportamento, non tanto da
  forzare la guida iper-difensiva. −2000 (40×) rischia la sovra-correzione: penalità > valore dell'intera rotta → ogni rischio di
  collisione diventa catastrofico → freeze/crawl → stuck+timeout risale (è il fallimento di D3, l'asse di H2 al contrario). −500 è un
  probe: se le collisioni scendono ma non basta → R3.1 a −1000; se −500 già sovra-corregge → soffitto trovato a basso costo.

  Sinergia con γ=0.997 (mantenuto): H2 dà l'orizzonte lungo, R3 dà la magnitudo. Insieme: penalità localmente decisiva (vince il
  trade-off vicino all'impatto, γ¹⁰≈0.97) e propagata all'indietro. R3 è il fix di magnitudo che a H2 mancava.

  Perché NON "ridurre lo shaping denso": la scala O(10^4) è creata dal +100/waypoint (sez. 1), il segnale primario di progresso. Ridurlo
  indebolisce l'incentivo a completare la rotta e cambia l'intera scala dei return → de-calibra H1/H1.1 (vf_clip/vf_loss_coeff tarati su
   questa scala) → confounded. Alzare il −50 è una riga, un numero, segnale di progresso intatto.

  Scheda R3

  - Edit (1 riga): carla_multi_agent_env.py:1748 — reward -= 50.0 → reward -= 500.0. Solo veicolo; il -50.0 pedone (:1841) resta (pedoni collidono ~0.1%).
  - Base: esito di H3 (H3 promosso → base H3; bocciato → 211055). Edit applicata solo al turno di R3.
  - Gate: vs base — SR +≥2.0, stuck+TO −≥2.0, collision/offroad ≤+1.0. Segnale di successo: collisione scende in modo netto. Spia di sovra-correzione: stuck+timeout (se risale, −500 è troppo).
  - Sicuro per il critic: −500 resta dentro vf_clip=1e6, nessun rischio NaN.

  (Aside, fuori da R3: l'off-road ha solo una penalità densa O(1), nessuna penalità terminale tipo il −50 della collisione — possibile candidato futuro, ma è un altro knob.)