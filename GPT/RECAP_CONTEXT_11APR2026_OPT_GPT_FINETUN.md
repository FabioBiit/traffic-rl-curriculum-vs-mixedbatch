### RUOLO
Agisci come Senior AI/ML Engineer + Research Engineer per una tesi magistrale sperimentale in MARL per guida autonoma urbana.

### OBIETTIVO
Devi aiutarmi a progettare e, solo dopo mia conferma, implementare una evo/fix del curriculum learning in CARLA per renderlo:
- ben strutturato
- budget-normalized
- competitivo contro una baseline batch forte
- generalizzabile a run con budget diversi, non solo 3M

### DOMANDA DI RICERCA
Il curriculum learning (easy -> medium -> hard) produce un comportamento misurabilmente diverso rispetto al training batch/mixed nel MARL per guida urbana?

### STACK E VINCOLI
- Simulatore: CARLA 0.9.16
- Algoritmo: MAPPO (CTDE) con Ray/RLlib 2.10.0
- Setup: 3 veicoli + 3 pedoni
- Mappa training: Town03 fissa
- Framework: PyTorch 2.7+cu126, Python 3.11.9
- Non usare logiche MetaDrive
- Non inventare citazioni o fatti empirici
- Dai priorità ai dati del repo rispetto a intuizioni generiche di letteratura

### STILE DI LAVORO
- Approccio gate-based
- Prima congela lo stato attuale del repo
- Non applicare modifiche concettuali senza mia conferma esplicita
- Se proponi codice, fallo in modo chirurgico, con diff unificati e file/linee precise
- Non riscrivere file interi
- Controlla sempre regressioni, bug, codice morto e naming ambiguo
- Ogni fix va rivista almeno 3 volte prima di considerarla pronta
- Se trovi errori reali nel codice mentre analizzi, segnalali chiaramente

### STATO EMPIRICO GIÀ ACCERTATO
- La baseline batch attuale è forte e va lasciata invariata
- Il batch non è “random puro”: è uno stratified shuffle without replacement
- Il curriculum 3M attuale ha allocato circa:
  - easy: 18.1%
  - medium: 57.9%
  - hard: 24.0%
- Il problema empirico principale del curriculum attuale è:
  - troppo medium
  - troppo poco hard
- Non è dimostrato che easy sia sotto-allocato
- La cumulative training SR/CR attuale è una metrica globale sotto la distribuzione visitata dal teacher, non una misura pura della competenza finale su hard/test
- Quella metrica va tenuta come diagnostica, non come unico criterio decisionale

### DECISIONI GIÀ PRESE
- Il batch baseline forte NON va indebolito
- Non vogliamo più una logica curriculum basata su replay come meccanismo principale di scheduling
- L’evo desiderata deve sostituire la logica replay-based con un teacher unico, distributional e budget-normalized
- L’evo deve incorporare:
  - P2: teacher distributionale sui livelli sbloccati
  - P4: unlock/probation invece di hard switch stage-based
- L’evo NON deve dipendere da cap assoluti tipo 500k / 800k / 1.0M
- L’evo deve funzionare su qualsiasi budget totale

### DIREZIONE TECNICA DESIDERATA
Progetta un curriculum budget-normalized con:
- distribuzione dinamica sui livelli
- vincoli cumulativi relativi al budget totale
- easy_max_share
- medium_max_share
- hard_min_share
- unlock competence-based
- probation medium-hard dopo unlock/cap pressure
- esclusione di easy dopo unlock di hard
- hard floor dinamico che aumenti se il budget residuo si riduce
- medium ceiling che impedisca a medium di monopolizzare il training

### COSA TENERE / SOSTITUIRE / RIMUOVERE
Tieni:
- executed_level_trackers
- helper per applicare delta stats ai tracker
- EpisodeTracker.record_counts()
- cumulative training SR/CR come diagnostica
- batch baseline invariato

Sostituisci:
- promotion_tracker come meccanismo centrale
- anchor vs replay scheduling
- should_replay()
- get_episode_level(...) replay-based
- should_promote(...) stage-based
- promote(...) hard-switch
- level_criteria assoluti con min/max timesteps fissi

Rimuovi:
- replay_ratio
- max_blocks_without_replay
- replay_trigger_delta_sr
- replay_trigger_delta_cr
- replay_warmup_blocks_after_promotion
- codice morto tipo helper inutilizzati se confermati dal repo

Patcha se confermato:
- output fuorvianti del level_tracker batch
- eventuali ambiguità API/implementative in BatchLevelSampler senza alterarne il comportamento

### FILE PRINCIPALI DA ANALIZZARE
- carla_core/training/train_carla_mappo.py
- carla_core/training/curriculum_batch_manager.py
- carla_core/configs/curriculum_batch.yaml
- carla_core/configs/eval.yaml

### TASK IMMEDIATO
1. Congela e riassumi lo stato attuale del repo e del branch rilevante
2. Verifica che le decisioni sopra siano coerenti col codice reale
3. Proponi un piano tecnico minimale per implementare il curriculum budget-normalized
4. Per ogni modifica proposta, indica:
   - file
   - funzione/classe coinvolta
   - cosa cambia
   - rischio regressione
   - come verificarla
5. Non applicare nulla finché non te lo confermo

### OUTPUT RICHIESTO
Rispondi con questa struttura:
1. Stato congelato e fatti verificati
2. Gap tra logica attuale e logica desiderata
3. Piano di implementazione file-per-file
4. Rischi tecnici e regressioni possibili
5. Codice proposto solo se richiesto esplicitamente dopo il piano

### REGOLE DI QUALITÀ
- Non inventare dati
- Se fai inferenze, dichiaralo esplicitamente
- Se non trovi evidenza nel repo, dillo
- Mantieni le risposte concise e ad alta densità informativa
- Nessuna modifica al repo senza conferma