"""
Smoke Test CARLA — Gate G1
==========================
Verifica che CARLA sia installato e funzionante.
Connette al server, carica Town03, spawna 5 veicoli + 5 pedoni con autopilot,
esegue 200 tick in modalità sincrona, stampa statistiche.

Prerequisiti:
    1. Server CARLA in esecuzione (CarlaUE4.exe)
    2. pip install carla==0.9.16

Esegui con:
    python carla_core/scripts/smoke_test_carla.py

Opzioni:
    --host 127.0.0.1   Host del server CARLA
    --port 2000         Porta TCP del server
    --map Town03        Mappa da caricare
    --vehicles 5        Numero di veicoli da spawnare
    --pedestrians 5     Numero di pedoni da spawnare
    --ticks 200         Numero di tick di simulazione
"""

import argparse
import sys
import time
import random

try:
    import carla
except ImportError:
    print("[ERRORE] Modulo 'carla' non trovato.")
    print("Installa con: pip install carla==0.9.16")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Smoke Test CARLA")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--vehicles", type=int, default=5)
    parser.add_argument("--pedestrians", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=200)
    args = parser.parse_args()

    spawned_vehicles = []
    spawned_walkers = []
    walker_controllers = []
    original_settings = None
    world = None
    client = None

    try:
        # ---- Connessione ----
        print(f"Connessione a {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        server_version = client.get_server_version()
        client_version = client.get_client_version()
        print(f"  Server: {server_version} | Client: {client_version}")

        if server_version != client_version:
            print(f"  [WARN] Version mismatch! Server={server_version}, Client={client_version}")

        # ---- Carica mappa ----
        available_maps = [m.split("/")[-1] for m in client.get_available_maps()]
        print(f"  Mappe disponibili: {', '.join(sorted(available_maps))}")

        print(f"  Caricamento {args.map}...")
        world = client.load_world(args.map)
        time.sleep(2.0)  # Attendi caricamento

        # ---- Modalità sincrona ----
        original_settings = world.get_settings()
        sync_settings = world.get_settings()
        sync_settings.synchronous_mode = True
        sync_settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(sync_settings)
        print("  Modalita' sincrona attivata (0.05s/tick)")

        # ---- Traffic Manager ----
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        print(f"  Spawn points disponibili: {len(spawn_points)}")

        if len(spawn_points) < args.vehicles:
            print(f"  [WARN] Solo {len(spawn_points)} spawn points, richiesti {args.vehicles}")
            args.vehicles = min(args.vehicles, len(spawn_points))

        # ---- Spawn veicoli ----
        print(f"\nSpawn {args.vehicles} veicoli...")
        vehicle_bps = blueprint_library.filter("vehicle.*")
        # Escludi bici e moto per semplicità
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels").as_int()) >= 4]

        random.shuffle(spawn_points)
        for i in range(args.vehicles):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            vehicle = world.try_spawn_actor(bp, spawn_points[i])
            if vehicle is not None:
                vehicle.set_autopilot(True, tm.get_port())
                spawned_vehicles.append(vehicle)
                print(f"  [{i+1}] {bp.id} → ID {vehicle.id}")
            else:
                print(f"  [{i+1}] Spawn fallito per {bp.id}")

        print(f"  Veicoli spawnati: {len(spawned_vehicles)}/{args.vehicles}")

        # ---- Spawn pedoni ----
        print(f"\nSpawn {args.pedestrians} pedoni...")
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = blueprint_library.find("controller.ai.walker")

        # Genera spawn points random per pedoni
        for i in range(args.pedestrians):
            spawn_loc = world.get_random_location_from_navigation()
            if spawn_loc is None:
                print(f"  [{i+1}] Nessun punto navigabile trovato")
                continue

            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            transform = carla.Transform(spawn_loc)
            walker = world.try_spawn_actor(bp, transform)
            if walker is not None:
                spawned_walkers.append(walker)
                print(f"  [{i+1}] {bp.id} → ID {walker.id}")
            else:
                print(f"  [{i+1}] Spawn fallito per {bp.id}")

        print(f"  Pedoni spawnati: {len(spawned_walkers)}/{args.pedestrians}")

        # Tick per registrare gli attori
        world.tick()

        # Spawn controller AI per ogni pedone
        for walker in spawned_walkers:
            controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
            if controller is not None:
                walker_controllers.append(controller)

        world.tick()

        # Avvia AI dei pedoni
        for controller in walker_controllers:
            target = world.get_random_location_from_navigation()
            if target is not None:
                controller.start()
                controller.go_to_location(target)
                controller.set_max_speed(1.4)  # ~5 km/h

        # ---- Simulazione ----
        print(f"\nEsecuzione {args.ticks} tick...")
        collisions = 0
        t_start = time.time()

        for tick in range(args.ticks):
            world.tick()

            if (tick + 1) % 50 == 0:
                # Heartbeat: stato veicoli
                active_v = sum(1 for v in spawned_vehicles if v.is_alive)
                active_w = sum(1 for w in spawned_walkers if w.is_alive)
                elapsed = time.time() - t_start
                print(f"  Tick {tick+1}/{args.ticks} | "
                      f"Veicoli: {active_v} | Pedoni: {active_w} | "
                      f"Elapsed: {elapsed:.1f}s")

        elapsed_total = time.time() - t_start
        sim_fps = args.ticks / elapsed_total if elapsed_total > 0 else 0

        # ---- Report ----
        print(f"\n{'='*50}")
        print("SMOKE TEST COMPLETATO")
        print(f"{'='*50}")
        print(f"  Mappa: {args.map}")
        print(f"  Veicoli spawnati: {len(spawned_vehicles)}")
        print(f"  Pedoni spawnati: {len(spawned_walkers)}")
        print(f"  Tick eseguiti: {args.ticks}")
        print(f"  Tempo reale: {elapsed_total:.1f}s")
        print(f"  Sim FPS: {sim_fps:.1f}")
        print(f"  Server: {server_version} | Client: {client_version}")
        print(f"\n[OK] CARLA funziona correttamente. Gate G1 (parte CARLA): PASS")

    except ConnectionRefusedError:
        print("[ERRORE] Connessione rifiutata. Il server CARLA e' in esecuzione?")
        print("  Avvia con: C:\\CARLA_0.9.16\\CarlaUE4.exe -quality-level=Low")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERRORE] Runtime: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        # ---- Cleanup ----
        print("\nCleanup...")
        for ctrl in walker_controllers:
            try:
                ctrl.stop()
                ctrl.destroy()
            except Exception:
                pass
        for walker in spawned_walkers:
            try:
                walker.destroy()
            except Exception:
                pass
        for vehicle in spawned_vehicles:
            try:
                vehicle.destroy()
            except Exception:
                pass

        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
            print("  Settings ripristinati.")

        print("  Cleanup completato.")


if __name__ == "__main__":
    main()
