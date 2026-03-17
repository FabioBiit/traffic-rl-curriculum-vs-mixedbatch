"""
GPU Check — Verifica che PyTorch e Ray usino la NVIDIA GPU
==========================================================
Uso:
    python carla_core/scripts/check_gpu.py
"""

import sys


def main():
    print("=" * 50)
    print("GPU CHECK")
    print("=" * 50)

    # PyTorch
    try:
        import torch
        print(f"\nPyTorch {torch.__version__}")
        print(f"  CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
            print(f"  Current device: {torch.cuda.current_device()} — {torch.cuda.get_device_name()}")

            # Quick test
            t = torch.randn(100, 100, device="cuda")
            print(f"  Tensor on CUDA: OK ({t.device})")
        else:
            print("  [WARN] CUDA non disponibile! Training usa CPU.")
            print("  Verifica: pip install torch --index-url https://download.pytorch.org/whl/cu126")
    except ImportError:
        print("  [ERRORE] PyTorch non installato")

    # Ray
    try:
        import ray
        ray.init(num_cpus=1, num_gpus=1, log_to_driver=False)
        resources = ray.available_resources()
        gpu_count = resources.get("GPU", 0)
        print(f"\nRay {ray.__version__}")
        print(f"  GPU rilevate da Ray: {gpu_count}")
        if gpu_count > 0:
            print(f"  [OK] Ray vede la GPU")
        else:
            print(f"  [WARN] Ray non vede GPU. Training su CPU.")
        ray.shutdown()
    except Exception as e:
        print(f"  [ERRORE] Ray: {e}")

    # NVIDIA SMI
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"\nnvidia-smi:")
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
        else:
            print("\n  [WARN] nvidia-smi fallito")
    except FileNotFoundError:
        print("\n  [WARN] nvidia-smi non trovato nel PATH")

    print(f"\n{'='*50}")


if __name__ == "__main__":
    main()
