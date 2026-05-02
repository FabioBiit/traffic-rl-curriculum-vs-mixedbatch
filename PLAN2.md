# Evo Route Robustness + Future Perception

## Summary

Separare la evo in due livelli per non bruciare comparabilita sperimentale:

1. **Evo A, consigliata subito:** correggere il route tracking quando un agente salta waypoint ma rientra sul percorso. Cambia reward/route progression, ma non cambia observation dims.
2. **Evo B, nuovo training dedicato:** aggiungere preview curve, hazard/path obstacles e percezione dinamica piu ricca. Cambia observation dims, quindi richiede nuova variante training.

## Evo A: Route Skip Recovery

- Implementare lookahead sui prossimi waypoint.
- Se l'agente rientra su un waypoint futuro, avanzare `current_wp_idx`.
- Penalizzare waypoint saltati invece di bloccare la route.
- Non cambiare `VEHICLE_OBS_DIM`, `PEDESTRIAN_OBS_DIM`, global critic dim o checkpoint compatibility.
- Metriche SR restano identiche: `termination_reason == "route_complete"`.

Configurazione proposta:

- `waypoint_lookahead: 8`
- `vehicle_waypoint_radius_m: 2.0`
- `pedestrian_waypoint_radius_m: 2.0`
- `skipped_waypoint_penalty: 10.0`

## Evo B: Curve + Path Hazard Awareness

Da fare solo come nuova variante sperimentale.

Aggiunte osservazionali:

- preview di 3 waypoint futuri;
- angolo curva e `curve_speed_hint`;
- hazard slots sul path davanti all'agente;
- TTC e lateral offset degli ostacoli lungo la traiettoria.

Comportamento desiderato:

- rallentare prima di curve strette;
- sterzare piu gradualmente;
- evitare agenti/NPC sul path;
- frenare solo in emergenza tramite safety guard.

Questa evo cambia observation space e richiede nuovo training. Va marcata nei risultati come variante separata, non confrontabile direttamente con `carla_mappo_20260430_153509`.

## Modifiche Consigliate

- Prima implementare Evo A e validarla in visualizzazione: e il fix piu mirato al problema osservato.
- Rimandare Evo B a un run successivo, dopo aver definito formalmente nuove obs dims e schema `observation.schema`.
- Aggiungere logging diagnostico opzionale:
  - `skipped_waypoints`;
  - `curve_severity`;
  - `min_path_ttc`;
  - `hazard_actor_type`.
- Salvare questi campi solo in `infos`/debug, non nelle metriche principali, finche non sono stabilizzati.
- Non modificare SR, collision rate, offroad rate o path efficiency nella prima iterazione.

## Test Plan

- Micro-test su route progression:
  - waypoint corrente raggiunto;
  - waypoint futuro raggiunto con skip;
  - agente fuori route senza advance;
  - route completata correttamente.
- Visual smoke:
  - hard level, follow vehicle;
  - verificare che l'agente non resti bloccato su waypoint saltato;
  - verificare che `route_complete` scatti solo a fine route.
- Regressione:
  - vecchi checkpoint devono restare caricabili con Evo A;
  - Evo B deve rifiutare checkpoint con obs dims vecchie o richiedere nuovo training.

## Assumptions

- Prima priorita: evitare stop/stuck artificiale da waypoint saltato.
- Collisione e offroad restano termination valide.
- Le feature di curva e avoidance ricca sono utili, ma vanno isolate in una nuova variante sperimentale.