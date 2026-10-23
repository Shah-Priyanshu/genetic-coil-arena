"""Check a GPU machine is ready for experiments: torch + CUDA, numba and the squad env.

    python scripts/gpu_smoke.py
"""

import json
import os
import platform
import time
from pathlib import Path


def main() -> None:
    report: dict = {"python": platform.python_version(), "machine": platform.node()}

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["device"] = torch.cuda.get_device_name(0)
            report["cuda"] = torch.version.cuda
            x = torch.randn(4096, 4096, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                x @ x
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            report["matmul_tflops_fp32"] = round(20 * 2 * 4096**3 / seconds / 1e12, 1)
    except ImportError:
        report["torch"] = None
    print(json.dumps(report, indent=2), flush=True)

    from genetic_coil_arena.envs.squad.runner import measure_throughput

    print("squad throughput:", measure_throughput(), flush=True)

    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:
        Path(run_dir, "smoke.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
