"""
Verifica Setup - Controlla che tutte le dipendenze siano installate.
Esegui con: python scripts/verify_setup.py
"""

import sys


def check(name, test_fn, optional=False):
    tag = "~" if optional else "X"
    try:
        result = test_fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [{tag}] {name}: {e}")
        return optional


def main():
    print("=" * 60)
    print("VERIFICA SETUP - Traffic RL Thesis")
    print("=" * 60)
    all_ok = True

    print("\n-- FASE 1: Prototipazione MetaDrive --")
    all_ok &= check("Python", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    all_ok &= check("PyTorch", lambda: __import__("torch").__version__)
    all_ok &= check("CUDA", lambda: f"{'Si - ' + __import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'No (CPU only)'}")
    all_ok &= check("Gymnasium", lambda: __import__("gymnasium").__version__)
    all_ok &= check("Stable-Baselines3", lambda: __import__("stable_baselines3").__version__)
    all_ok &= check("MetaDrive", lambda: __import__("metadrive").__version__ if hasattr(__import__("metadrive"), "__version__") else "installato")
    all_ok &= check("NumPy", lambda: __import__("numpy").__version__)
    all_ok &= check("Matplotlib", lambda: __import__("matplotlib").__version__)
    all_ok &= check("TensorBoard", lambda: __import__("tensorboard").__version__)
    all_ok &= check("W&B", lambda: __import__("wandb").__version__)
    all_ok &= check("PyYAML", lambda: __import__("yaml").__version__ if hasattr(__import__("yaml"), "__version__") else "installato")
    all_ok &= check("tqdm", lambda: __import__("tqdm").__version__)

    print("\n-- FASE 2: CARLA + RLlib (necessario da settimana 3) --")
    check("Ray", lambda: __import__("ray").__version__, optional=True)
    check("RLlib", lambda: (__import__("ray.rllib.algorithms.ppo", fromlist=["PPOConfig"]), "OK")[1], optional=True)
    check("CARLA", lambda: __import__("carla").__version__, optional=True)
    check("PettingZoo", lambda: __import__("pettingzoo").__version__, optional=True)

    print("\n-- OPZIONALE: GNN (solo se gate G5) --")
    check("PyTorch Geometric", lambda: __import__("torch_geometric").__version__, optional=True)

    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] Setup base completato! Puoi procedere.")
        print("\nProssimo step:")
        print("  python training/train_metadrive.py")
    else:
        print("[!!] Alcune dipendenze mancano. Rivedi l'installazione.")
    print("=" * 60)


if __name__ == "__main__":
    main()