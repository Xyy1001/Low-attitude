from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "build_panel.py",
    "run_main.py",
    "run_robustness.py",
    "run_placebo.py",
    "run_simulation.py",
    "run_extended_experiments.py",
    "run_external_validation.py",
    "make_figures.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script}", flush=True)
        subprocess.run(
            [sys.executable, "-X", "utf8", str(ROOT / "experiments" / script)],
            cwd=ROOT,
            check=True,
        )
    print("Core real-data and simulation pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
